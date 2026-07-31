import time 
import math 
import queue
import torch 
import torch.nn.functional as f 

class RoutingTelemetry:
    def __init__(self,num_experts=8,top_k=2):
        self.num_experts=num_experts
        self.top_k=top_k
        self.queue=queue.Queue(maxsize=2000)
        self._hooks=[]
        self._step_accumulator={}
        self._step_token_count={}
        self._total_layers=0
        self._last_token_time=None
        self._token_times=[]        # rolling window for smoothed tok/s
        self._session_expert_totals=[0.0]*num_experts
        self._session_tokens=0
        self.active=False

    def attach(self,model):
        layers=model.base_model.model.layers
        self._total_layers=len(layers)    
        for i, layer in enumerate(layers):
            router=layer.lora_moe_block.router

            def make_hook(layer_idx):
                def hook(module,inputs,outputs):
                    if not self.active:
                        return 
                    # output is (routed_tensor, logits)
                    # logits: [num_tokens, num_experts]
                    _, logits=outputs 
                    with torch.no_grad():
                        weights=f.softmax(logits.float(),dim=-1)
                        top_k_indices=torch.topk(weights,self.top_k,dim=1).indices
                        counts=torch.zeros(self.num_experts)
                        for exp_idx in range(self.num_experts):
                            counts[exp_idx]=(top_k_indices==exp_idx).sum().item()
                    self._step_accumulator[layer_idx]=counts.tolist()
                    self._step_token_count[layer_idx]=logits.shape[0]
                    # when all 36 experts have fired that is one forward pass 
                    if len(self._step_accumulator)==self._total_layers:
                        self.emit()

                return hook 
            h=router.register_forward_hook(make_hook(i))
            self._hooks.append(h)        
        print(f"  Telemetry: attached hooks to {self._total_layers} router layers")

    def emit(self):
        # sum xpert counts over all the layers 
        total_per_expert=[0.0]*self.num_experts
        for layer_idx,counts in self._step_accumulator.items():
            for exp_idx,c in enumerate(counts):
                total_per_expert[exp_idx]+=c

        # compute per-layer expert assignment for this token
        layer_experts = []
        for layer_idx in sorted(self._step_accumulator.keys()):
            counts = self._step_accumulator[layer_idx]
            expert_id = max(range(self.num_experts), key=lambda i: counts[i])
            layer_experts.append(expert_id)

        self._step_accumulator.clear()
        self._step_token_count.clear()

        total=sum(total_per_expert)
        expert_pct=[c/total*100 if total>0 else 0 for c in total_per_expert]

        # routing entropy: 0 = fully fucked . log2(8)=3 means perfectly balanced 
        probs=[c/total for c in total_per_expert] if total>0 else [1/8]*8
        entropy=-sum(p*math.log2(p+1e-9) for p in probs)
        entropy_pct=entropy/math.log2(self.num_experts)*100  # 0-100%

        # tok/s — rolling 10-sample window
        now=time.monotonic()
        if self._last_token_time is not None:
            self._token_times.append(now - self._last_token_time)
            if len(self._token_times)>10:
                self._token_times.pop(0)
        self._last_token_time=now
        avg_interval=sum(self._token_times)/len(self._token_times) if self._token_times else 1.0
        toks_per_sec=round(1.0/avg_interval,1) if avg_interval>0 else 0.0

        # session cumulative
        self._session_tokens+=1
        for i in range(self.num_experts):
            self._session_expert_totals[i]+=expert_pct[i]
        session_avg=[t/self._session_tokens for t in self._session_expert_totals]

        payload = {
            "type": "routing",
            "expert_pct": [round(p, 1) for p in expert_pct],
            "session_avg": [round(p, 1) for p in session_avg],
            "entropy_pct": round(entropy_pct, 1),
            "tokens_per_sec": toks_per_sec,
            "token_index": self._session_tokens,
            "layer_experts": layer_experts,
        }
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            pass  # dashboard fell behind — drop the frame, don't block generation

    def reset_session(self):
        self._session_expert_totals=[0.0]*self.num_experts
        self._session_tokens=0
        self._token_times=[]
        self._last_token_time=None
        # drain stale data from queue
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except queue.Empty: break

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()     
