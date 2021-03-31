from eval.vqa.gqa_eval import GQAEval
from eval.vqa.plot_tail import plot_tail_for_one_model
import argparse
import numpy as np
import os.path
import glob
import json
from option import ROOT

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default=f'{ROOT}/GQA')
parser.add_argument('--eval_tail_size', action='store_true')
parser.add_argument('--ood_test', action='store_true')
parser.add_argument('--predictions', type=str)
args = parser.parse_args()


def loadFile(name):
    # load standard json file
    if os.path.isfile(name):
        with open(name) as file:
            data = json.load(file)
    # load file chunks if too big
    elif os.path.isdir(name.split(".")[0]):
        data = {}
        chunks = glob.glob('{dir}/{dir}_*.{ext}'.format(dir=name.split(".")[0], ext=name.split(".")[1]))
        for chunk in chunks:
            with open(chunk) as file:
                data.update(json.load(file))
    else:
        raise Exception("Can't find {}".format(name))
    return data


if args.eval_tail_size:
    result_eval_file = args.predictions

    # Retrieve scores
    alpha_list = [9.0, 7.0, 5.0, 3.6, 2.8, 2.2, 1.8, 1.4, 1.0, 0.8, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4,
                  -0.5, -0.6, -0.7]
    acc_list = []
    for alpha in alpha_list:
        ques_file_path = f'{args.data_root}/val_bal_tail_%.1f.json' % alpha
        predictions = loadFile(result_eval_file)
        # predictions = {p["questionId"]: p["prediction"] for p in predictions}

        gqa_eval = GQAEval(predictions, ques_file_path, choices_path=None, EVAL_CONSISTENCY=False)
        acc = gqa_eval.get_acc_result()['accuracy']
        acc_list.append(acc)

    print("Alpha:", alpha_list)
    print("Accuracy:", acc_list)
    # Plot: save to "tail_plot_[model_name].pdf"
    # plot_tail(alpha=list(map(lambda x: x + 1, alpha_list)), accuracy=acc_list,
    #           model_name='default')  # We plot 1+alpha vs. accuracy
elif args.ood_test:
    result_eval_file = args.predictions
    file_list = {'Tail': 'ood_testdev_tail.json', 'Head': 'ood_testdev_head.json', 'All': 'ood_testdev_all.json'}
    result = {}
    for setup, ques_file_path in file_list.items():
        predictions = loadFile(result_eval_file)
        # predictions = {p["questionId"]: p["prediction"] for p in predictions}

        gqa_eval = GQAEval(predictions, f'{args.data_root}/' + ques_file_path, choices_path=None,
                           EVAL_CONSISTENCY=False)
        result[setup] = gqa_eval.get_acc_result()['accuracy']

        result_string, detail_result_string = gqa_eval.get_str_result()
        print('\n___%s___' % setup)
        for result_string_ in result_string:
            print(result_string_)

    print('\nRESULTS:\n')
    msg = 'Accuracy (tail, head, all): %.2f, %.2f, %.2f' % (result['Tail'], result['Head'], result['All'])
    print(msg)
# Sample command:
# python eval/vqa/gqa_eval_from_file.py --predictions "{EXPT_ROOT}/gqa_1.1/SpectralDecouplingTrainer/lambda_0.001_gamma_0.001/ans_preds_Val All_Main_epoch_30.json" --eval_tail_size
