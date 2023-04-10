import pandas as pd
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D, Dataset
import os
from accelerate import Accelerator
from transformers import set_seed
import datasets
import transformers
from transformers import LayoutLMv2ForSequenceClassification
import torch
from transformers import AdamW
#from tqdm.notebook import tqdm
from tqdm import tqdm
from accelerate import DistributedDataParallelKwargs

df1=pd.read_csv('./DocTypeFinal2.csv')
#check=[1005,8111,3011,2180,5020,2183,3357,8001,2074,2078,3145,10021,7007,2051,10046,7015,2047,3169,2101,3242,8216]
check=[1,2]
result1=df1[df1['Priority'].isin(check)]
labels=result1['DocType'].tolist()
print(len(labels))

sample=os.listdir('./new_data/OnlyP2')
# sample=sample[:7]
# sample.remove('data-401')
# sample.remove('data-402')
sample
okay=[]
for i in sample:
         try:
            um = Dataset.load_from_disk('new_data/OnlyP2/' + str(i))
            print("done")
            okay.append(um)
         except Exception as e:
                print(e)
                continue
        
        
#train=okay[:29]
#valid=okay[29:34]
train=okay
print(len(train))
import datasets
train_encoded_dataset=datasets.concatenate_datasets(train)

"""
Based on the scenario, remove and rename the columns appropriately.
"""
# train_encoded_dataset=train_encoded_dataset.remove_columns(['path','super-labels'])
# train_encoded_dataset=train_encoded_dataset.rename_column('document-type','labels')

#labels
id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}


def create_dataloader(encoded_dataset_0,batch_size=4):
    encoded_dataset_0.set_format(type="torch",device="cuda")
    dataloader = torch.utils.data.DataLoader(encoded_dataset_0, batch_size=4,shuffle=True)
    return dataloader


hyperparameters = {
    "learning_rate": 5e-5,
    "num_epochs": 20,
    "batch_size": 2, # Actual batch size will this x 8
    "seed": 39,
}

def training_function():
    #put the model in training mode
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    dataloader=create_dataloader(train_encoded_dataset,batch_size=hyperparameters['batch_size'])
    set_seed(hyperparameters["seed"])
    model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", 
                                                            num_labels=len(labels))
    
    # The next two lines of code need to be uncommented if you are continuing training from a saved checkpoint
#     model.load_state_dict(torch.load('./saved-model/P1-20-final.pth',map_location='cpu'))
#     model = model.cuda()
#    accelerator.print("Model loaded successfully")
    optimizer = AdamW(model.parameters(), lr=hyperparameters['learning_rate'],)
#     # Uncomment the next two lines if you have a saved checkpoint of the optimizer.
#     optimizer.load_state_dict(torch.load('./saved-model/P1-20-final-optimiser.pth',map_location='cpu'))
#     accelerator.print("Optimiser loaded successfully")
    model, optimizer,dataloader = accelerator.prepare(model, optimizer,dataloader)
    num_train_epochs=hyperparameters['num_epochs']
    global_step = 0
    t_total = len(dataloader) * num_train_epochs
    try:
        for epoch in range(num_train_epochs):
          model.train()
          accelerator.print("Epoch:", epoch)
          running_loss = 0.0
          correct = 0
          for batch in tqdm(dataloader):
              # forward pass
              outputs = model(**batch)
              loss = outputs.loss

              running_loss += loss.item()
              predictions = outputs.logits.argmax(-1)
              correct += (predictions == batch['labels']).float().sum()

              # backward pass to get the gradients 
              #loss.backward()
              accelerator.backward(loss)


              # update
              optimizer.step()
              optimizer.zero_grad()
              global_step += 1

          accelerator.print("Loss:", running_loss / batch["input_ids"].shape[0])
          accuracy = 100 * correct / train_encoded_dataset.num_rows
          accelerator.print("Training accuracy:", accuracy.item())

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), './saved-model/P1P2_4Nov.pth')
        accelerator.save(optimizer.state_dict(),  "./saved-model/P1P2_4Nov_optimiser.pth")
        
        # The next set of commented codes are different ways of saving the model.
        #unwrapped_model.save_pretrained(
        #    "./um/11", save_function=accelerator.save
        #)
        #accelerator.save(
         #   {
                #"epoch": completed_epochs,
                #"steps": completed_steps,
         #       "optimizer": optimizer.state_dict(),
                #"scheduler": lr_scheduler.state_dict(),
                #"scaler": accelerator.scaler.state_dict(),
          #  },
           # "./um/9-optim",
        #)
    except Exception as e:
        print(e)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), './saved-model/P1P2_4Nov.pth')
        accelerator.save(optimizer.state_dict(),  "./saved-model/P1P2_4Nov_optimiser.pth")
        #unwrapped_model.save_pretrained(
         #   "./um/11", save_function=accelerator.save
        #)
        #accelerator.save(
         #   {
                #"epoch": completed_epochs,
                #"steps": completed_steps,
          #      "optimizer": optimizer.state_dict(),
                #"scheduler": lr_scheduler.state_dict(),
                #"scaler": accelerator.scaler.state_dict(),
           # },
            #"./um/9-optim",
        #)
        
from accelerate import notebook_launcher

notebook_launcher(training_function,num_processes=4,use_fp16=False)
