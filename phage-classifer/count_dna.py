import numpy as np
import pyfastx
import phanotate_modules.functions as phano
shotsetds=[]
shotsetss=[]
fa_ds = pyfastx.Fasta('ds-linear.fasta')
fa_ss = pyfastx.Fasta('ss-circular.fasta')
def shot(str):
    length=len(str)
    start=np.random.randint(length,10)
    return str[start:start+1500]
for itm in fa_ds:
    for _ in range(10):
        shotsetds.append(shot(itm.seq))
for itm in fa_ss:
    for _ in range(10):
        shotsetss.append(shot(itm.seq))
shotsetds=[phano.get_backgroud_rbs(i) for i in shotsetds]
# shotsetds = torch.Tensor(shotsetds)
# shotsetds = torch.Tensor.reshape(shotsetds,(-1,28))
shotsetss=[phano.get_backgroud_rbs(i) for i in shotsetss]
# shotsetss = torch.Tensor(shotsetss)
# shotsetss = torch.Tensor.reshape(shotsetss,(-1,28))
shotsetlist=shotsetds+shotsetss
shotsetlist=np.array(shotsetlist,dtype=np.float32)
labels1=np.ones(544)
labels2=np.zeros(484)
label=[labels1,labels2]
label=np.r_[labels1,labels2]
label=np.array(label,dtype=np.int)
from sklearn.model_selection import train_test_split
#x为数据集的feature熟悉，y为label.
x_train, x_test, y_train, y_test = train_test_split(shotsetlist, label, test_size = 0.2,random_state=4)
shotset=CustomDataset(torch.Tensor(x_train),y_train)
myloader=data.DataLoader(shotset,batch_size=15)
testset=CustomDataset(torch.Tensor(x_test),y_test)
testloader=data.DataLoader(shotset,batch_size=15)
d2l.train_ch3(model, myloader, testloader, loss, num_epochs, trainer)