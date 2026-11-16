"""LoRA fine-tuning of the flow network on the reference's tutorial dataset (~50 PLINDER systems).

    uv run --extra convert --extra train --extra cuda python examples/finetune_lora.py

The dataset is `test_dataset/test_dataset.tar.gz` of the Hugging Face repo `nvidia/RFD4-Proteina`, unpacked into
`$DATA_PATH/test_dataset` (`DATA_PATH` defaults to `~/.cache/jrfd4_proteina`). Any reference training config
works: pass its Hydra overrides to `convert.train_loader`.
"""

import argparse
import time

import jax

from jrfd4_proteina import convert, io, lora, optim, training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--out", default="adapters")
    args = parser.parse_args()

    model = convert.load("nvidia/RFD4-Proteina").bfloat16()  # frozen; bf16 mixed precision, as the reference trains
    config = training.release_training_config()
    examples = convert.train_loader(
        [
            "experiment=tutorial/finetune_test_dataset",
            f"dataloaders.batch_size={args.batch_size}",
            "dataloaders.num_workers=4",  # featurize while the device works on the previous step
            "dataloaders.prefetch_factor=2",
            "dataloaders.multiprocessing_context=spawn",
        ]
    )

    key = jax.random.key(0)
    adapters = lora.init(model.flow, key, rank=32, alpha=64.0)
    optimizer = optim.optimizer(args.learning_rate)
    opt_state = optimizer.init(adapters)
    average = adapters

    start = time.perf_counter()
    for step, batch in zip(range(args.steps), examples):
        adapters, opt_state, losses = training.train_step(
            adapters, opt_state, model, optimizer, batch, jax.random.fold_in(key, step), config
        )
        average = training.ema(average, adapters)
        if step % 10 == 0:
            print(
                f"step {step:5d}  loss {losses.total:8.3f}  bb_ca {losses.bb_ca:6.3f}  "
                f"latents {losses.local_latents:7.3f}  sequence {losses.sequence:5.3f}  "
                f"{time.perf_counter() - start:6.1f} s"
            )

    io.save(average, args.out)  # reload with io.load; sample with lora.merge(model.flow, adapters)


if __name__ == "__main__":
    main()
