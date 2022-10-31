import torch.nn.functional as F
import torch
from tqdm import tqdm


def adaptive_learning_phase(model, dataset, optimizer, criterion, hc_s, hc_ns, ct_s, ct_ns, use_tqdm=True):
   cnt_update = [0, 0]

   acc_arr = []
   sen_arr = []
   total_loss = 0

   update_batch_x, update_batch_y = [], []
   
   model.eval()
   cnt_correct, cnt_total = 0, 0
   cnt_correct_seizure = 0
   for x, y in tqdm(dataset, disable=not use_tqdm):
      outputs = model(x.view(1, -1).type(torch.FloatTensor)).detach()
      pred = torch.max(outputs, axis=1)[1].item()
      softmax_output = F.softmax(outputs, dim=1)

      loss = criterion(outputs, y.view(1))
      total_loss += loss.item()

      if softmax_output[0, y] >= [ct_ns, ct_s][y]:
         if len(update_batch_y) != 0 and update_batch_y[0] != y:
            update_batch_x = []
            update_batch_y = []
         update_batch_x.append(x)
         update_batch_y.append(y)
      else:
        update_batch_x = []
        update_batch_y = []
      
      if len(update_batch_y) >= [hc_ns, hc_s][y]:
         x_train = update_batch_x[-1].unsqueeze(0)
         y_train = update_batch_y[-1].unsqueeze(0)

         # model.train()
         optimizer.zero_grad()
         outputs = model(x_train.type(torch.FloatTensor))
         loss = criterion(outputs, y_train)
         loss.backward()
         optimizer.step()

         cnt_update[y] += 1
      
      cnt_total += 1
      if pred == y:
         cnt_correct += 1
         if y == 1:
            cnt_correct_seizure += 1

      acc_arr.append(cnt_correct/cnt_total)
      sen_arr.append(cnt_correct_seizure/dataset[:][1].sum())
   
   total_loss = total_loss / len(dataset)

   print("Count of update:", cnt_update)
   
   return acc_arr, sen_arr, total_loss
