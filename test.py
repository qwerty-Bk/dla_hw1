import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_asr.model as module_model
from hw_asr.datasets.utils import get_dataloaders
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.trainer import Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.metric.cer_metric import calc_cer
from hw_asr.metric.wer_metric import calc_wer

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # text_encoder
    language = config.get('lang', 'en')
    if language == 'en':
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
    elif language == 'ru':
        text_encoder = CTCCharTextEncoder.get_russian_alphabet()
    else:
        raise ValueError(f'Incorrect language. Supported languages: en, ru. Got: {language}')

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    results = []

    wer_sum, cer_sum = 0, 0
    wer_bs_sum, cer_bs_sum = 0, 0

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)
            output = model(**batch)
            if type(output) is dict:
                batch.update(output)
            else:
                batch["logits"] = output
            batch["log_probs"] = torch.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = model.transform_input_lengths(
                batch["spectrogram_length"]
            )
            batch["probs"] = batch["log_probs"].exp().cpu()
            batch["argmax"] = batch["probs"].argmax(-1)
            for i in range(len(batch["text"])):
                argmax = batch["argmax"][i]
                argmax = argmax[:int(batch["log_probs_length"][i])]
                probs = batch["log_probs"][i]
                probs = probs[:int(batch["log_probs_length"][i])]
                pred_text_argmax = text_encoder.ctc_decode(argmax.tolist())
                pred_text_bs = text_encoder.ctc_beam_search(probs)
                wer_argmax = calc_wer(batch["text"][i], pred_text_argmax)
                cer_argmax = calc_cer(batch["text"][i], pred_text_argmax)
                wer_bs = calc_wer(batch["text"][i], pred_text_bs[0][0])
                cer_bs = calc_cer(batch["text"][i], pred_text_bs[0][0])
                results.append(
                    {
                        "ground_trurh": batch["text"][i],
                        "pred_text_argmax": pred_text_argmax,
                        "pred_text_beam_search": pred_text_bs[:10],
                        "wer_argmax": wer_argmax,
                        "cer_argmax": cer_argmax,
                        "wer_bs": wer_bs,
                        "cer_bs": cer_bs
                    }
                )
                wer_sum += wer_argmax
                cer_sum += cer_argmax
                wer_bs_sum += wer_bs
                cer_bs_sum += cer_bs
    with Path(out_file).open("w") as f:
        json.dump({
            "final_wer": wer_sum / len(results),
            "final_cer": cer_sum / len(results),
            "final_wer_beam_search": wer_bs_sum / len(results),
            "final_cer_beam_search": cer_bs_sum / len(results)
        }, f, indent=2)
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": str(test_data_folder / "audio"),
                            "transcription_dir": str(
                                test_data_folder / "transcriptions"
                            ),
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output)
