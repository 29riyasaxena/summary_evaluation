import BARTScore
from BARTScore.bart_score import BARTScorer
from blanc import BlancHelp, BlancTune
import evaluate
import os


def evaluate(summary_path, references_path):
    orignal = []
    for filename in os.listdir(references_path):
      with open(os.path.join(references_path, filename), 'r') as f:
        orignal.append(f.read())

    predicted=[]
    for filename in os.listdir(summary_path):
      with open(os.path.join(summary_path, filename), 'r') as f:
        predicted.append(f.read())
    
    blanc_help = BlancHelp(device='cuda')
    blanc_tune = BlancTune(device='cuda', finetune_mask_evenly=False, show_progress_bar=False)
    blanc_score = blanc_help.eval_pairs(orignal, predicted)

    meteor = evaluate.load('meteor')
    meteor_pred = []
    for i in range(len(orignal)):
      meteor_pred.append(meteor.compute(predictions=[predicted[i]], references=[orignal[i]]))
   
    rouge = evaluate.load('rouge')
    rouge_pred = []
    for i in range(len(orignal)):
      rouge_pred.append(rouge.compute(predictions=[predicted[i]], references=[orignal[i]]))

    bertscore = load("bertscore")
    bertscore_pred = []
    for i in range(len(orignal)):
      bertscore_pred.append(bertscore.compute(predictions=[predicted[i]], references=[orignal[i]],lang="en"))

    meteor = evaluate.load('meteor')
    meteor_pred = []
    for i in range(len(orignal)):
      meteor_pred.append(meteor.compute(predictions=[predicted[i]], references=[orignal[i]]))
  
    chrf = evaluate.load("chrf")
    chrf_pred = []
    for i in range(len(orignal)):
      chrf_pred.append(chrf.compute(predictions=[predicted[i]], references=[orignal[i]]))
  
    bleu = evaluate.load("bleu")
    bleu_pred = []
    for i in range(len(orignal)):
      bleu_pred.append(bleu.compute(predictions=[predicted[i]], references=[orignal[i]]))
  
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
    bart_pred = bart_scorer.score(predicted, orignal)

    data = {"blanc": blanc_score, "meteor": meteor_pred , "rouge": rouge_pred, "bertscore":bertscore_pred, "chrf": chrf_pred,"bleu" :bleu_pred, "bartscore": bart_pred}
    df_evaluation = pd.DataFrame(data=data)  
    return df_evaluation.to_csv(eval.csv)

if __name__ == '__main__':
  summary_path = input("Enter Summary Path: ")
  references_path = input("Enter References Path: ")
  evaluate(summary_path, references_path)