package org.deeplearning4j.logAnalysis.lstm;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Log data Iterator
 * @author Liucy
 *
 */
public class LogDataIterator {
	private static final int VECTOR_SIZE = 4;
    //data items for every batch
    private int batchNum;

    //data length for every batch
    private int exampleLength;

    //store all the log data
    private List<LogBean> dataList;

    //save the beginning no. of training items for each batch.
    private List<Integer> dataRecord;

    private int[] maxNum={10,3,50,19};

    public LogDataIterator(){
        dataRecord = new ArrayList<>();
    }

    /**
     * load data and initialize
     * */
    public boolean loadData(String fileName, int batchNum, int exampleLength){
        this.batchNum = batchNum;
        this.exampleLength = exampleLength;
//        maxNum = new int[VECTOR_SIZE];
        //load log data items
        try {
            readDataFromFile(fileName);
        }catch (Exception e){
            e.printStackTrace();
            return false;
        }
        
        resetDataRecord();
        return true;
    }

    /**
     * reset the training batch record
     * */
    private void resetDataRecord(){
        dataRecord.clear();
        int total = dataList.size()/exampleLength+1;
		for (int i = 0; i < total; i++) {
			if (i * exampleLength != dataList.size())
				dataRecord.add(i * exampleLength);
		}
    }

    /**
     * read data items from file.
     * */
    public List<LogBean> readDataFromFile(String fileName) throws IOException{
        dataList = new ArrayList<>();
        FileInputStream fis = new FileInputStream(fileName);
        BufferedReader in = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
        in.readLine(); //skip the first line (title)
        String line = in.readLine();
/*        for(int i=0;i<maxNum.length;i++){
            maxNum[i] = 0;
        }*/
        System.out.println("Reading data items..");
        while(line!=null){
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
        return dataList;
    }

    public int[] getMaxArr(){
        return this.maxNum;
    }

    public void reset(){
        resetDataRecord();
    }

    public boolean hasNext(){
        return dataRecord.size() > 0;
    }

    public DataSet next(){
        return next(batchNum);
    }

    /**
     * Build training items for the next iteration
     * */
    long prevDataTime=0;
    public DataSet next(int num){
        if( dataRecord.size() <= 0 ) {
            throw new NoSuchElementException();
        }
        int actualBatchSize = Math.min(num, dataRecord.size());
//        System.out.println(exampleLength+" - "+dataList.size()+" - "+dataRecord.get(0));
        int actualLength = Math.min(exampleLength,dataList.size()-dataRecord.get(0)-1);
        INDArray input = Nd4j.create(new int[]{actualBatchSize,VECTOR_SIZE,actualLength}, 'f');
        INDArray label = Nd4j.create(new int[]{actualBatchSize,4,actualLength}, 'f');
        LogBean nextData = null,curData = null;
        //Fetch the features and labels of every batch.
        for(int i=0;i<actualBatchSize;i++){
            int index = dataRecord.remove(0);
            int endIndex = Math.min(index+exampleLength,dataList.size()-1);
            curData = dataList.get(index);
            for(int j=index;j<endIndex;j++){
                //Take the next data item as label
                nextData = dataList.get(j+1);
                //create the features
                long timeIntv=curData.getTime()-prevDataTime;
                if(timeIntv<0)
                {
                	System.out.println("Error timestamp, reverse order!!");
                	timeIntv=-timeIntv;
                }
                input.putScalar(new int[]{i, 0, j-index}, (double)timeIntv/maxNum[0]);
                input.putScalar(new int[]{i, 1, j-index}, (double)curData.getHostID()/maxNum[1]);
                input.putScalar(new int[]{i, 2, j-index}, (double)curData.getProgramID()/maxNum[2]);
                input.putScalar(new int[]{i, 3, j-index}, (double)curData.getSeverity()/maxNum[3]);
                //create the label(s)
                label.putScalar(new int[]{i, 0, j-index}, (double)(nextData.getTime()-curData.getTime())/maxNum[0]); // take the time difference as label
                label.putScalar(new int[]{i, 1, j-index}, (double)nextData.getHostID()/maxNum[1]);
                label.putScalar(new int[]{i, 2, j-index}, (double)nextData.getProgramID()/maxNum[2]);
                label.putScalar(new int[]{i, 3, j-index}, (double)nextData.getSeverity()/maxNum[3]);

                prevDataTime=curData.getTime();
                curData = nextData;
            }
            if(dataRecord.size()<=0) break;
        }
        return new DataSet(input, label);
    }

    public int batch() {
        return batchNum;
    }

    public int cursor() {
        return totalExamples() - dataRecord.size();
    }

    public int numExamples() {
        return totalExamples();
    }

    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public int totalExamples() {
        return (dataList.size()) / exampleLength;
    }

    public int inputColumns() {
        return dataList.size();
    }

    public int totalOutcomes() {
        return 1;
    }
}
