import numpy as np
import matplotlib.pyplot as plt

def sygma(x):
	return 1/(1+np.exp(-x))
	
def prim_sygma(x):
	return x*(1-x)	
	
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'].astype('float32')/255, f['y_train']
        
        y_train=np.eye(10)[y_train]
        x_test, y_test = f['x_test'].astype('float32')/255, f['y_test']
        return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data('D:/Downloads/mnist.npz')
x_train2=[]
for i in range(len(x_train)):
	x_train2.append(x_train[i].reshape(1,784)[(0)])
		
layers_sizes=[784,20,10]
count_layers=len(layers_sizes)
layers=[]
epochs=3
step_learning=0.01

matrix_of_weights=[np.random.uniform(-0.5,0.5,(layers_sizes[i],layers_sizes[i-1])) for i in range(1,count_layers)]

matrix_of_bias=[np.zeros((layers_sizes[i],1)) for i in range(1,count_layers)]

layers=[np.zeros((layers_sizes[i],1)) for i in range(count_layers)]

for i in range(epochs):
	e_correct=0
	for image,label in zip(x_train2,y_train):
		
		layers[0]=image.reshape((-1,1))
		label=label.reshape((-1,1))
		
		for k in range(1,count_layers):
			layers[k]=sygma(matrix_of_weights[k-1]@ layers[k-1]+matrix_of_bias[k-1])
			
		err=2*(layers[-1]-label)
		
		e_correct+=int(np.argmax(layers[-1])==np.argmax(label))
		
		for k in range(count_layers-2,-1,-1):
		   	  	
		   	matrix_of_weights[k]-=step_learning*err @ np.transpose(layers[k])
		   	
		   	matrix_of_bias[k]-=step_learning*err
		   	
		   	err=np.transpose(matrix_of_weights[k])@ err * prim_sygma(layers[k])
		   	
	print(round((e_correct/len(x_train))*100,3))	   	

test=plt.imread('test.bmp')
gray=lambda rgb : np.dot(rgb[... , :3],[0.299,0.587,0.114])
test=gray(test).astype("float32")/255

image=test.reshape(-1)
image=np.reshape(test, (-1,1))

layers[0]=image.reshape((-1,1))
for k in range(1,count_layers):
	layers[k]=sygma(matrix_of_weights[k-1]@ layers[k-1]+matrix_of_bias[k-1])

print(layers[-1].argmax())



