def smoke_test_loader(loader_cls, tokenizer, context_length=512, n_samples=5):
    print(f"\n{'='*50}")
    print(f"Smoke test: {loader_cls.__name__}")
    print(f"{'='*50}")

    loader = loader_cls(tokenizer=tokenizer, context_length=context_length)
    batch  = loader.collect(n_samples=n_samples)

    actual = len(batch["input_ids"])
    assert actual > 0, "No samples collected!"

    for i in range(actual):
        ids    = batch["input_ids"][i]
        labels = batch["labels"][i]

        assert len(ids) == len(labels),        f"Sample {i}: input_ids/labels length mismatch"
        assert len(ids) <= context_length,      f"Sample {i}: exceeds context length ({len(ids)})"
        assert any(l != -100 for l in labels),  f"Sample {i}: all labels masked"

        n_label_toks = sum(1 for l in labels if l != -100)
        print(f"  Sample {i}: {len(ids)} tokens | {n_label_toks} label tokens unmasked")

    # ── ADD THIS BLOCK ────────────────────────────────────────────────
    if hasattr(loader, "lang_counts") and loader.lang_counts:
        total = sum(loader.lang_counts.values())
        print(f"\n  Language breakdown (format-passed, n={total}):")
        for lang, count in sorted(loader.lang_counts.items()):
            print(f"    {lang:<12}: {count:>5}  ({count/total*100:.1f}%)")
    # ─────────────────────────────────────────────────────────────────

    print(f"\n  PASSED — {actual}/{n_samples} samples valid\n")


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

    from leetcode_loader import LeetCodeLoader as x

    smoke_test_loader(
                    x
                      , tokenizer)
