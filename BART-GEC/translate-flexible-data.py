import torch
import argparse
from fairseq.models.bart import BARTModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        help='Path to the model directory', required=True)
    parser.add_argument('--input_text',
                        help='Path to the input text', required=True)
    parser.add_argument('--output_dir',
                        help='Path to the output dir', required=True)
    parser.add_argument('--checkpoint_file',
                        help='checkpoint name', required=True)
    parser.add_argument('--data_path',
                        help='checkpoint name', required=True)
    args = parser.parse_args()
    model_dir = args.model_dir
    input_text = args.input_text
    output_path = args.output_dir
    data_path = args.data_path
    bart = BARTModel.from_pretrained(
        model_dir,
        checkpoint_file=args.checkpoint_file,
        data_name_or_path=data_path
    )

    bart.cuda()
    bart.eval()
    bart.half()
    count = 1
    bsz = 128
    with open(input_text) as source, open(output_path, 'w') as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in source:
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch = bart.sample(slines, beam=1)

                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []

            slines.append(sline.strip())
            count += 1
        if slines != []:
            hypotheses_batch = bart.sample(slines, beam=1)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
    print("fin")
