package org.deeplearning4j.logAnalysis.lstm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import org.deeplearning4j.logAnalysis.WaveletActivation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class LogLstmPredict {
	private static final int IN_NUM = 4;
	private static final int OUT_NUM = 4;
	private static final int Epochs = 20;
	  
	private static final int lstmLayer1Size = 32;
	private static final int lstmLayer2Size = 64;
	
	private static final int predictSteps = 10;
	  
	public static MultiLayerNetwork getNetModel(int nIn,int nOut, String waveletID){
	    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
	        .learningRate(0.001) //0.1
	        .rmsDecay(0.85)
	        .seed(12345)
	        .regularization(true)
	        .l2(0.001)
	        .weightInit(WeightInit.XAVIER)
	        .updater(Updater.RMSPROP).momentum(0.9)	//RMSPROP
	        .list()
	        .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(lstmLayer1Size)  
	            .activation(new WaveletActivation(waveletID)).build())	//.activation(new WaveletActivation(waveletID)).build())
	        .layer(1, new GravesLSTM.Builder().nIn(lstmLayer1Size).nOut(lstmLayer2Size)
	            .activation(Activation.TANH).build())
	        .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
	            .nIn(lstmLayer2Size).nOut(nOut).build())
	        .pretrain(false).backprop(true)
//	        .backpropType(BackpropType.TruncatedBPTT)
//	        .tBPTTForwardLength(50) //50-200
//	        .tBPTTBackwardLength(50)
	        .build();
	  
	    MultiLayerNetwork net = new MultiLayerNetwork(conf);  
	    net.init();  
	    net.setListeners(new ScoreIterationListener(1));  //set monitor, print score every iteration step
	  
	    return net;  
	}
	
	public static void train(MultiLayerNetwork net,LogDataIterator iterator, INDArray[] initArrayArr, FileOutputStream fos){
		//Iteration training
		for(int i=0;i<Epochs;i++) {
			DataSet dataSet = null;
			while (iterator.hasNext()) {
				dataSet = iterator.next();
				net.fit(dataSet);
			}
			iterator.reset();
			// record log
			try {
				fos.write(("Finished the number "+(i+1)+" round training!!\r\n").getBytes());
				test(net, iterator, initArrayArr, fos);//////////
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			
			net.rnnClearPreviousState();
		}
	}
	
	public static void test(MultiLayerNetwork net,LogDataIterator iterator, INDArray[] initArrayArr, FileOutputStream fos) throws IOException
	{
		for(int i=0; i<initArrayArr.length; i++)
		{
		INDArray initArray=initArrayArr[i];
		INDArray output = net.rnnTimeStep(initArray);
		//record log
		fos.write(("For tesing item "+(i+1)+" , The predicted result is: \r\n").getBytes());
		
		for(int j=0;j<predictSteps;j++) {
			int[] maxNums = iterator.getMaxArr();
			fos.write((output.getDouble(0)*maxNums[0]+"    "+output.getDouble(1)*maxNums[1]+"    "+output.getDouble(2)*maxNums[2]+"    "+output.getDouble(3)*maxNums[3]+"\r\n").getBytes());
			output = net.rnnTimeStep(initArray);
		}
		}
	}

	private static INDArray[] getTestingArray(LogDataIterator iter) throws IOException{
		int TESTING_ITEM_SIZE=1; // the number of data items to pick for predicting
		int testItems=TESTING_ITEM_SIZE; //Read the number of items from testing data set
        List<LogBean> dataList = new ArrayList<>();
        String inputFile = LogLstmPredict.class.getClassLoader().getResource("logAnalysis/logLstm/logData_verify.csv").getPath();
        FileInputStream fis = new FileInputStream(inputFile);
        BufferedReader in = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
        in.readLine(); //skip the first line (title)
        String line = in.readLine();
        System.out.println("Reading verifying data items..");
        while(line!=null && testItems-->0){
            String[] strArr = line.split(",");
            if(strArr.length>=3) { //////////////////////
            	LogBean data = new LogBean();
                //build data item
                data.setTime(Long.valueOf(strArr[0]));
                data.setHostID(Integer.valueOf(strArr[1]));
                data.setProgramID(Integer.valueOf(strArr[2]));
                data.setSeverity(Integer.valueOf(strArr[3]));
                dataList.add(data);
            }
            line = in.readLine();
        }
        in.close();
        fis.close();
        // instantiate each element to INDArray
        INDArray[] indArr=new INDArray[TESTING_ITEM_SIZE];
        long prevDataTime=0;
        for(int i=0; i<indArr.length; i++)
        {
        	int[] maxNums = iter.getMaxArr();
        	LogBean lb=dataList.get(i);
        	indArr[i] = Nd4j.zeros(1, 4, 1); //1 dimension, every element is 3x1 matrix 
        	indArr[i].putScalar(new int[]{0,0,0}, (double)(lb.getTime()-prevDataTime)/maxNums[0]);
        	indArr[i].putScalar(new int[]{0,1,0}, (double)lb.getHostID()/maxNums[1]);
        	indArr[i].putScalar(new int[]{0,2,0}, (double)lb.getProgramID()/maxNums[2]); 
        	indArr[i].putScalar(new int[]{0,3,0}, (double)lb.getSeverity()/maxNums[3]);
        	
        	prevDataTime=lb.getTime();
        }
		return indArr;
	}
	
	public static void main(String[] args) {
	    String inputFile = LogLstmPredict.class.getClassLoader().getResource("logAnalysis/logLstm/logData.csv").getPath();
	    int batchSize = 1;
	    int exampleLength = 16;
	    //Initialize the Neural Network
	    LogDataIterator iterator = new LogDataIterator();
	    iterator.loadData(inputFile,batchSize,exampleLength);
	  
	    try {
			INDArray[] initArrayArr = getTestingArray(iterator);
			FileOutputStream fos=new FileOutputStream(new File("d:/LogAnalysis.txt"));
			
			boolean USE_WAVELET=true;
			
			if(USE_WAVELET)
			{
			// Add wavelets to model
	        String[] waveletArr={ "Haar","Daubechies 2","Daubechies 3","Daubechies 4","Daubechies 5","Daubechies 6","Daubechies 7","Daubechies 8","Daubechies 9","Daubechies 10","Daubechies 11","Daubechies 12","Daubechies 13","Daubechies 14","Daubechies 15","Daubechies 16","Daubechies 17","Daubechies 18","Daubechies 19","Daubechies 20","Coiflet 1","Coiflet 2","Coiflet 3","Coiflet 4","Coiflet 5","Symlet 2","Symlet 3","Symlet 4","Symlet 5","Symlet 6","Symlet 7","Symlet 8","Symlet 9","Symlet 10","Symlet 11","Symlet 12","Symlet 13","Symlet 14","Symlet 15","Symlet 16","Symlet 17","Symlet 18","Symlet 19","Symlet 20","BiOrthogonal 1/1","BiOrthogonal 1/3","BiOrthogonal 1/5","BiOrthogonal 3/1","BiOrthogonal 3/3","BiOrthogonal 3/5","BiOrthogonal 3/7","BiOrthogonal 3/9"};
	        for(String waveletID : waveletArr)
	        {
	        	fos.write(("************************************************************\r\n").getBytes());
	        	fos.write(("Begin the network model of: "+waveletID+"\r\n").getBytes());
	        	MultiLayerNetwork net = getNetModel(IN_NUM,OUT_NUM, waveletID);
				train(net, iterator, initArrayArr, fos);
				fos.write(("End the network model of: "+waveletID+"\r\n").getBytes());
				fos.write(("************************************************************\r\n").getBytes());
	        }
			}else{
				MultiLayerNetwork net = getNetModel(IN_NUM,OUT_NUM, "");
				train(net, iterator, initArrayArr, fos);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	} 
}
