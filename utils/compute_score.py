import torch
from torch.autograd import Variable

def compute_score(model, data_set, batch_size=1000, cuda=False):
  dtype=torch.FloatTensor
  if cuda:
    dtype = torch.cuda.FloatTensor

  # --- sequential loader
  data_loader = torch.utils.data.DataLoader(
                        data_set,
                        batch_size=batch_size,
                        shuffle=False)

  # --- test on data_loader
  correct = 0
  correct_discrete = 0
  for data, target in data_loader:
    data = Variable(data.type(dtype))
    target = Variable(target.type(dtype).long())
    output = model(data)
    pred = output.data.max(1, keepdim=True)[1]
    correct += float(pred.eq(target.data.view_as(pred)).sum())
    if (pred == pred[0]).all():
      print("all samples predicted to same class.")

    output = model(data, discrete=True)
    pred = output.data.max(1, keepdim=True)[1]
    correct_discrete += float(pred.eq(target.data.view_as(pred)).sum())
  score = correct/len(data_loader.dataset)
  score_discrete = correct_discrete/len(data_loader.dataset)
  
  return score, score_discrete


