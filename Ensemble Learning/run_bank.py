import csv
import math
from os import error

TRAIN_FILE='./bank/train.csv'
TEST_FILE='./bank/test.csv'

'''
1. Title: Bank Marketing

2. Relevant Information:

   The data is related with direct marketing campaigns of a Portuguese banking institution. 
   The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, 
   in order to access if the product (bank term deposit) would be (or not) subscribed. 

   The classification goal is to predict if the client will subscribe a term deposit (variable y).

3. Number of Attributes: 16 + output attribute.

4. Attribute information:

   Input variables:
   # bank client data:
   1 - age (numeric)
   2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services") 
   3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
   4 - education (categorical: "unknown","secondary","primary","tertiary")
   5 - default: has credit in default? (binary: "yes","no")
   6 - balance: average yearly balance, in euros (numeric) 
   7 - housing: has housing loan? (binary: "yes","no")
   8 - loan: has personal loan? (binary: "yes","no")
   # related with the last contact of the current campaign:
   9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
  10 - day: last contact day of the month (numeric)
  11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
  12 - duration: last contact duration, in seconds (numeric)
   # other attributes:
  13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
  14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
  15 - previous: number of contacts performed before this campaign and for this client (numeric)
  16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

  Output variable (desired target):
  17 - y - has the client subscribed a term deposit? (binary: "yes","no")
'''


columns=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','label']
attributes=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']
numeric=['age','balance','day','duration','campaign','pdays','previous']

ATTRIBUTES,MST=None,None

def init_attr():
    global columns,ATTRIBUTES,MST
    dataset={}
    anal={}
    reader=csv.reader(open(TRAIN_FILE,'r'))
    for c in columns:
        if c==columns[-1]: # ignore "label"
            continue
        dataset[c]=[]
        anal[c]={}
    for sample in reader:
        for i in range(len(columns)-1):
            if sample[i] not in dataset[columns[i]]:
                dataset[columns[i]].append(sample[i])
                anal[columns[i]][sample[i]]=0
            anal[columns[i]][sample[i]]+=1
    mst={}
    for k in anal.keys():
        st=sorted(anal[k].items(), key=lambda x: x[1], reverse=True)
        mm=st[0][0]
        if mm=='unknown':
            mm=st[1][0]
        mst[k]=mm
    ATTRIBUTES,MST = dataset,mst


def def_data(labels):
    dic={}
    for label in labels:
        if label not in dic.keys():
            dic[label]={}
    return dic

def init_data(reader, filt=0):
    global columns,MST
    dataset=[]
    for sample in reader:
        atrs={}
        for i in range(len(columns)):
            if columns[i] not in atrs.keys():
                mm=sample[i]
                if filt==1:
                    if mm=='unknown':
                        mm=MST[columns[i]]
                atrs[columns[i]]=mm
        dataset.append(atrs)
    return dataset

def get_subset(S,attr,value):
    dataset=[]
    for sample in S:
        if sample[attr]==value:
            res={}
            for label in sample.keys():
                if label==attr:
                    continue
                res[label]=sample[label]
            dataset.append(res)
    return dataset

def Log(a):
    return -a*math.log2(a)      
        
def Entropy(dic,sum):
    res=0
    for key in dic.keys():
        if key=='sum':
            continue
        res+=Log(dic[key]/sum)
    return res

def ME(dic,sum):
    return sorted(dic.items(), key=lambda x: x[1])[0][1]/sum

def GI(dic,sum):
    res=1
    for key in dic.keys():
        if key=='sum':
            continue
        res-=(dic[key]/sum)**2
    return res

def Gain(entropy,attribute,data,algo='Entropy'):
    gain=entropy
    for value in data[attribute].keys():
        if value=='sum':
            continue
        if algo=='Entropy':
            nbr=Entropy(data[attribute][value],data[attribute][value]['sum'])
        elif algo=='ME':
            nbr=ME(data[attribute][value],data[attribute][value]['sum'])
        elif algo=='GI':
            nbr=GI(data[attribute][value],data[attribute][value]['sum'])
        gain-=(data[attribute][value]['sum']/data['sum'])*nbr
    return gain

class Node:
    def __init__(self,label) -> None:
        self.data=label
        self.children={}

def analyse(S):
    global columns
    col=[]
    for label in S[0].keys():
        col.append(label)
    attr=def_data(col)
    attr['sum']=0

    for sample in S:
        attr['sum']+=1
        for label in col:
            if label==columns[-1]:
                if sample[label] not in attr[label].keys():
                    attr[label][sample[label]]=0
                attr[label][sample[label]]+=1
            else:
                if sample[label] not in attr[label].keys():
                    attr[label][sample[label]]={'sum':0}
                attr[label][sample[label]]['sum']+=1
                if sample[columns[-1]] not in attr[label][sample[label]].keys():
                    attr[label][sample[label]][sample[columns[-1]]]=0
                attr[label][sample[label]][sample[columns[-1]]]+=1
    return attr
    
def get_best_attr(attr,algo='Entropy'):
    col=[]
    for label in attr.keys():
        if label!='sum':
            col.append(label)
    result={}
    for a in col:
        if a=='label':
            continue
        if algo=='Entropy':
            result[a]=Gain(Entropy(attr['label'],attr['sum']),a,attr,algo='Entropy')
        elif algo=='ME':
            result[a]=Gain(ME(attr['label'],attr['sum']),a,attr,algo='ME')
        elif algo=='GI':
            result[a]=Gain(GI(attr['label'],attr['sum']),a,attr,algo='GI')
    return sorted(result.items(), key=lambda x: x[1], reverse=True)[0][0]

MAX_LAY=0

def ID3(S,Attributes,layer,max_layer=0,algo='Entropy'):
    global MAX_LAY
    if layer>MAX_LAY:
        MAX_LAY=layer
    label_dic={}
    for sample in S:
        if sample['label'] not in label_dic.keys():
            label_dic[sample['label']]=0
        label_dic[sample['label']]+=1
    label_sorted=sorted(label_dic.items(), key=lambda x: x[1], reverse=True)
    if len(label_dic.keys())==1:
        return Node(label_sorted[0][0])
    if len(Attributes)==0:
        return Node(label_sorted[0][0])
    if max_layer>0 and layer>max_layer:
        return Node(label_sorted[0][0])
    
    A=get_best_attr(analyse(S),algo)
    #print(A)
    n=Node(A)
    for value in ATTRIBUTES[A]:
        Sv=get_subset(S,A,value)
        #print(Sv)
        if len(Sv)==0:
            n.children[value]=Node(label_sorted[0][0])
        else:
            Attr=[]
            for a in Attributes:
                if a!=A:
                    Attr.append(a)
            n.children[value]= ID3(Sv,Attr,layer+1,max_layer)
        #print(A+':'+value+'->'+n.children[value].data)
    return n

def get_val(dic,key):
    if key in dic.keys():
        return dic[key]
    else:
        minn=9999
        mink=''
        for k in dic.keys():
            dif=abs(int(k)-int(key))
            if dif<minn:
                minn=dif
                mink=k
        return dic[mink]


def predict(root,sample):
    if len(root.children)==0:
        return root.data
    else:
        return predict(get_val(root.children,sample[root.data]),sample)

def test(root, S):
    corr=0
    summ=0
    for sample in S:
        summ+=1
        y=predict(root,sample)
        if y==sample[columns[-1]]:
            corr+=1
    #print('acc: '+str(corr/summ))
    print('err: '+str((summ-corr)/summ))


def run_test(train_data,test_data):
    for i in range(1,17):
        print('layer='+str(i))
        print('Entropy: ',end='')
        test(ID3(train_data,attributes,1,i,'Entropy'),test_data)
        print('ME: ',end='')
        test(ID3(train_data,attributes,1,i,'ME'),test_data)
        print('GI: ',end='')
        test(ID3(train_data,attributes,1,i,'GI'),test_data)

def main():
    init_attr()
    train_data_reader=csv.reader(open(TRAIN_FILE,'r'))
    test_data_reader=csv.reader(open(TEST_FILE,'r'))

    # the 'unknown' values will be replace when the second parameter is 1
    # the 'unknown' values will be retained when the second parameter is 0
    train_data=init_data(train_data_reader,1)
    test_data=init_data(test_data_reader,1)

    print('run test on testing data:')
    run_test(train_data,test_data)
    print('run test on training data:')
    run_test(train_data,train_data)

if __name__ == '__main__':
    main()

def sign(n):
    if n > 0:
        return 1
    else:
        return -1

class Adaboost:
    def __init__(self,X,Y) -> None:
        self.D=[]
        self.m=0
        self.t=0
        self.x=[]
        self.y=[]
        self.A=[]
        self.H=[]

        self.x=X
        self.y=Y
        self.m=len(X)
        self.D=[1/self.m for i in range(self.m)]
    
    def error(self,h):
        s=0
        for i in range(self.m):
            s+=self.D[i]*self.y[i]*h(self.x[i])
        return 0.5-0.5*s
    
    def vote(self,e):
        return 0.5*math.log(((1-e)/e),math.e)

    def update_D(self,h):
        e=self.error(h)
        a=self.vote(e)
        Z=0
        for i in range(self.m):
            self.D[i]=self.D[i]*math.exp(-a*self.y[i]*h(self.x[i]))
            Z+=self.D[i]
        for i in range(self.m):
            self.D[i]=self.D[i]/Z
        self.A.append(a)
        self.H.append(h)
        self.t+=1
    
    def predict(self,x):
        s=0
        for i in range(self.t):
            s+=self.A[i]*self.H[i](x)
        return sign(s)