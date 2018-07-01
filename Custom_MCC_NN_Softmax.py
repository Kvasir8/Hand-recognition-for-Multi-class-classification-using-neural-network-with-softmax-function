import cv2
import tensorflow as tf
import numpy as np
import random #mini batch를 사용하기 위해서는 Random한 data set을 묶음으로 batch학습을 시켜야하므로 필요하다.
import matplotlib.pyplot as plt #실제 data value에 대한 정보를 matlab의 plot처럼 graph를 도식화하기 위해 불러온다.
import Kmeans
import Image_data_loader as IDL #image data를 불러와서 train & test dat set으로 만듦

#from tensorflow.examples.tutorials.mnist import input_data
# MNIST data set을 읽어들이게 된다. 
#이는 0~9까지 행과 각각 hand-craft로 쓰여진 무수히 많은 갯수(=n)의 열을 갖으므로, 총 20*n = 20n개의 MNIST data set을 읽어들인다.

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 상위 폴더에 있는 directory주소에 대해 data set을 읽어들이고 mnist라는 변수에 저장한다.
#one_hot이 true이므로, 각 train data set마다 배열요소의 순서를 숫자로 할당시키는 역할을 한다.
#즉, 다음 아래 코드행부터 mnist를 쓸 때 마다 one hot으로 읽어들이게 된다.

#batch_size = 100 #mini batch형태로 몇 개의 묶음 단위로 epoch할지 정한다.

    #parameters 파라메터값들
training_epochs = 60 #전체 data set을 한번 돌을 epoch 횟수를 지정한다.
nb_classes = 3 # K = 3개인 Multi Variable Problem 이므로 Label의 갯수는 3개가 된다.
Input_Image_data_size = 100*100
N_HW = Input_Image_data_size #100*100 = 10000개의 Hidden weights 갯수 값을 지정한다.
trainLoss=[]
testLoss=[]
learning_rate = 0.001
keep_prob = tf.placeholder(tf.float32) # dropout을 설정하기 위한 변수

    # Input image data : 100 * 100 = 10000
X = tf.placeholder(tf.float32, [None, Input_Image_data_size]) #None은 n개 원하는 만큼 data를 입력을 줄 수 있다.
Y = tf.placeholder(tf.float32, [None, nb_classes]) #K = 3이므로 3개의 출력을 지정한다.

    #Hidden layer's weights
W1 = tf.Variable(tf.random_normal([Input_Image_data_size, N_HW])) #입력을 10000개 data씩 받는다.
W2 = tf.Variable(tf.random_normal([N_HW, N_HW]))
W3 = tf.Variable(tf.random_normal([N_HW, N_HW]))
W4 = tf.Variable(tf.random_normal([N_HW, N_HW]))
W5 = tf.Variable(tf.random_normal([N_HW, nb_classes])) #최종 출력(0~9)인 10개에 대응되는 weight 갯수를 지정한다.
    #bias values
B1 = tf.Variable(tf.random_normal([Input_Image_data_size])) #Hidden layer weight 갯수에 상응된 값으로 bias를 지정한다.
B2 = tf.Variable(tf.random_normal([N_HW])) #Hidden layer weight 갯수에 상응된 값으로 bias를 지정한다.
B3 = tf.Variable(tf.random_normal([N_HW])) #Hidden layer weight 갯수에 상응된 값으로 bias를 지정한다.
B4 = tf.Variable(tf.random_normal([N_HW])) #Hidden layer weight 갯수에 상응된 값으로 bias를 지정한다.
B5 = tf.Variable(tf.random_normal([nb_classes])) #Label은 3개이므로 bias값 또한 상응되게 3개로 지정한다.

    #Layer model
HL1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
HL1 = tf.nn.dropout(HL1, keep_prob = keep_prob)

HL2 = tf.nn.relu(tf.add(tf.matmul(HL1, W2), B2))
HL2 = tf.nn.dropout(HL2, keep_prob = keep_prob)

HL3 = tf.nn.relu(tf.add(tf.matmul(HL2, W3), B3))
HL3 = tf.nn.dropout(HL3, keep_prob = keep_prob)

HL4 = tf.nn.relu(tf.add(tf.matmul(HL3, W4), B4))
HL4 = tf.nn.dropout(HL4, keep_prob = keep_prob)

hypothesis = tf.add(tf.matmul(HL4, W5), B5) #where h = wx + b < Hypothesis > 

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = Y)) #softmax hypothesis
#cross_entropy_with_logits(hypothesis, Y)에서 hypothesis로 들어오는 값들이 softmax로 취하지 않은 값을 의미하며,
#with_logits은 logits말그대로 들어오는 입력값(z)을 softmax의 h(z)를 모두 더한다는 의미이다.
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #앞서 얻어진 cost값을 Gradient Descent를 한다.
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) #앞서 얻어진 cost값을 AdamOptimizer한다.
   
with tf.Session() as sess: 
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer()) 
    #앞서 선언한 global변수들을 재사용하기 위해variables_initializer를 실행시켜 weight변수 값들을 초기화시켜준다.
    
    for epoch in range(training_epochs): #epoch가 1~지정해준 epoch횟수 까지 learning 하게 된다.
        Tr_cost = 0 #cost값을 저장하기위한 변수를 선언한다.
        Te_cost = 0
        total_batch = 10
        #offset = 1000000000
        # 이러한 100개의 mini batch size를 통해 iteration을 몇 번 학습 시킬지 정해야한다.
        #전체 batch_size의 갯수를 전체 mnist 20n개의 dat set으로 나눠주어 적절한 iteration이 되도록 한다.
        
        for i in range(total_batch):
            IDL.shuffle()
            batch_xs_Tr = IDL.Select('Mini_X_Tr')
            batch_ys_Tr = IDL.Select('Mini_Y_Tr')
            batch_xs_Te = IDL.Select('Mini_X_Te')
            batch_ys_Te = IDL.Select('Mini_Y_Te')
            
            # mnist 변수를 읽어온 data set에 대하여 100개 data set batch형태로 x와 y가 읽어진다.
            #이는 mnist의 총 data set은 10000개 이상이므로 한꺼번에 불러오면 epoch당 연산량이 불필요하게 많아지게 되는 것을 방지한다.
            #즉, 1epoch당 100개의 data set이 호출되어 연산된다.
            # Mini Batch : next_batch(batch_size)의 구조는 batch_size = 100개씩 처음 batch_xs/ys에 할당하고,
            #그 다음 101~200 = 100개씩 다시한번 batch_xs/ys에 할당해준다.
            #이러한 방식으로 iteration을 새로운 batch set이 100개씩 data set이 들어오게 된다.
            Tr_loss, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs_Tr, Y: batch_ys_Tr, keep_prob: 0.7 })
            Te_loss = sess.run(cost, feed_dict={ X: batch_xs_Te, Y: batch_ys_Te, keep_prob: 0.7 })
            #feed_dict는 앞서 초기화 시킨 data set값을 넣어준다는 의미이다.
             
            # 그리고 iteration마다 새롭게 들어오게되는 새로운 100개 batch는 feed_dict에 의해 X와 Y에 값이 할당이 되고
            #sess.run에 의해 이전에 정의해준 cost에 의해 hypothesis가 softmax를 거치게 되고 label값 들이 update된다. 
            #또한 optimizer를 통해 learning rate에 따라 Adam Optimizer가 실행이 되고 cost값이 최종적으로 낮아지게 된다.
            Tr_cost += Tr_loss / (528*(100*100)) #528 training data set 100x100 image data elements 50 batch size
            Te_cost += Te_loss / (90*(100*100)) #90 test data set
        
            
        trainLoss.append(Tr_cost)
        testLoss.append(Te_cost)
        print('Epoch:', '%2d' % (epoch + 1), 'Train_cost =', '{:.2f}'.format(Tr_cost), 'Test_cost =', '{:.2f}'.format(Te_cost))
        #epoch & 연산 된 cost를 실시간으로 표시한다.
  
    
    print("-----Done Learning-----")
    #학습이 완전하게 끝나게 됨. 밑의 내용은 학습된 내용을 검증하기 위한 test dat set과 비교하는 과정이다.
    
    #print(cost1)  #printing final loss
    print("Test epoch corresponding to min :", np.argmin(testLoss)) #몇 번째 epoch가 최소값을 만족했는지를 표시
    # finding the epoch that corresponds to min.
    print("its value :", testLoss[np.argmin(testLoss)]) #몇 번째 epoch가 최소값을 만족했는지를 표시
    # finding the epoch that corresponds to min.
    
    #TrainLoss에 해당되는 graph를 표시한다.
    plt.figure(0)
    plt.plot(trainLoss)
    plt.plot(testLoss)
    
    # Test the model using test sets; Y label값과 Hidden layer를 거친 최종적인 hypothesis값이 얼마나 유사한지 계산한다.
    pred_correction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(pred_correction, tf.float32))
    #이렇게 hypothesis값으로 나온 값과 Y label값을 비교하여 나온 값을 float32형으로 바꾸어주어 형변환을 통해 연산을 수월하게 해준다.
    #부동소수점으로 cast된 값들을 모두 평균내어 accuracy에 할당.

    AC = sess.run(accuracy, feed_dict = { X: IDL.Select('X_Te'), Y: IDL.Select('Y_Te'), keep_prob: 1 })
    #AC = accuracy.eval(session=sess, feed_dict={ X: mnist.test.images, Y: mnist.test.labels }) 
    #k_p = 1은 모든 network을 총 동원한다는 의미.
    print("Accuracy: %.4f" %(AC*100))
    # 학습에 사용되지 않은 test data set을 불러와서 Labal으로 지정한다.
    # 해당코드에는 one_hot이 활성화됐으므로 Y: mnist.test.labels를 one_hot으로 읽어들이게 된다. 즉, 별도로 one_hot으로 만들지 않아도
    #읽어들일 때, one_hot으로 처리한다.
    #추가적으로 ~.eval()은 sess.run(~)과 동일한 기능을 한다. eval()명령어기능을 통해 Accuracy를 console창에 띄울 수 있게 된다.

    # Get one and predict
    #r_char = random.randint(0, mnist.test.num_examples - 1)
    #총 0~9개의 classes를 갖으며 0~9의 숫자 중 무작위 정수integer숫자하나에 해당하는 one_hot으로 되어 있는 mnist data set을 불러온다.
    plt.figure(1) #Test image를 띄우기 위한 또 다른 plot figure를 생성한다.
    plt.show() #해당 plot figure를 화면에 띄운다.
    
    while 1: 
        cmd = input('Menu : ')
        if(cmd == 'i'):            
            #openCV
            fn = input('Enter your image file : ')
            mask = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2GRAY)
            Pred = sess.run(tf.argmax(hypothesis, 1), feed_dict={X : mask.reshape(1,-1), keep_prob: 1})
            print("Prediction: ", Pred)
            #1개의 data scalar값을 갖는 hypothesis값읇 불러오고, 그 값에 맞는 Image 정보값을 feed_dict를 통해 dictionary형으로 받아온다.
            plt.figure(2) #Test image를 띄우기 위한 또 다른 plot figure를 생성한다.
            plt.imshow(mask, cmap='Greys')
            plt.show() #해당 plot figure를 화면에 띄운다.
            # r_char에 의해 test image값이 정해진 것을 mnist image data를 원본 mnist data image는 큰 image형태이므로 28x28=784 image 형태로
            #reshape시켜 축소시키고 회색조grey형태로 나타내어 plot하여 graph를 도식화한다.
        elif(cmd == 'k'):
            Enb = 1
            fn = 'K_hand.jpg'
            src = cv2.imread(fn)
            
            src = cv2.resize(src, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            
            
            lower = np.array((0,38,0))
            upper = np.array((30,255,255))
            mask = cv2.inRange(hsv, lower, upper)
            Km = mask
            
            ret, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            total_contours = len(contours)
            if (Enb == 1):
                Km[Km == 255] = 1
                Km = np.transpose(np.nonzero(Km))
                #Kmeans.Result(Km, total_contours, 5)
                Kmeans.Result(Km, 3, 10)

            '''
            while 1:
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                src = frame
                hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
                
                lower = np.array((0,28,0))
                upper = np.array((40,255,255))
                mask = cv2.inRange(hsv, lower, upper)
                Km = mask
                
                ret, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                total_contours = len(contours)
                
                Km[Km == 255] = 1
                Km = np.transpose(np.nonzero(Km))
                Kmeans.Result(Km, total_contours, 5)
                
                cv2.imshow('mask',src)
                cv2.imshow('mask',mask)
                
                cv2.waitKey(200) #200ms delay
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
                '''
        elif(cmd == 'c'):
            cap = cv2.VideoCapture(0)
            while 1:
                ret, frame = cap.read()
                W, H,_ = frame.shape
                src = frame
                
                Gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                mask = Gray[W//5+30 : W//5+300+30, H//5-30 : H//5+300-30]
                mask = cv2.resize(mask, None, fx=0.333, fy=0.333, interpolation=cv2.INTER_CUBIC) 
                #100x100 Image transformation
                cv2.imshow('mask',mask) 
                Pred = sess.run(tf.argmax(hypothesis, 1), feed_dict={X : mask.reshape(1,-1), keep_prob: 1})
                text = "Recognition : ["+ str(Pred) +"]"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(src,text,(15,30), font, .5, (255, 255, 255), 2)
                cv2.rectangle(src,(W//5,H//5),(W//5+300,H//5+300),(0,255,0),3)
                cv2.imshow('src',src)
                    
                cv2.waitKey(100) #100ms delay
        
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
            
        elif(cmd == 'q'):
            break
