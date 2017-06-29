package org.deeplearning4j.logAnalysis.lstm.uci;

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
public class AirQualityDataIterator {
	private static final int VECTOR_SIZE = 1;
    //data items for every batch
    private int batchNum;

    //data length for every batch
    private int exampleLength;

    //store all the log data
    private List<Float> dataList;

    //save the beginning no. of training items for each batch.
    private List<Integer> dataRecord;

    private float[] maxNum={1};

    public AirQualityDataIterator(){
        dataRecord = new ArrayList<>();
    }

    /**
     * load data and initialize
     * */
    public boolean loadData(String fileName, int batchNum, int exampleLength){
        this.batchNum = batchNum;
        this.exampleLength = exampleLength;
//        maxNum = new float[VECTOR_SIZE];
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
    public List<Float> readDataFromFile(String fileName) throws IOException{
        dataList = new ArrayList<>();
        FileInputStream fis = new FileInputStream(fileName);
        BufferedReader in = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
        in.readLine(); //skip the first line (title)
        String line = in.readLine();

        System.out.println("Reading data items..");
        while(line!=null){
			float val = Float.parseFloat(line);
//			if (Math.abs(val) > maxNum[0]) maxNum[0] = Math.abs(val);
			// build data item
			dataList.add(val);
			line = in.readLine();
        }
        in.close();
        fis.close();
        return dataList;
    }

    public float[] getMaxArr(){
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
    public DataSet next(int num){
        if( dataRecord.size() <= 0 ) {
            throw new NoSuchElementException();
        }
        int actualBatchSize = Math.min(num, dataRecord.size());
//        System.out.println(exampleLength+" - "+dataList.size()+" - "+dataRecord.get(0));
        int actualLength = Math.min(exampleLength,dataList.size()-dataRecord.get(0)-1);
        INDArray input = Nd4j.create(new int[]{actualBatchSize,VECTOR_SIZE,actualLength}, 'f');
        INDArray label = Nd4j.create(new int[]{actualBatchSize,1,actualLength}, 'f');
        float curData, nextData;
        //Fetch the features and labels of every batch.
        for(int i=0;i<actualBatchSize;i++){
            int index = dataRecord.remove(0);
            int endIndex = Math.min(index+exampleLength,dataList.size()-1);
            curData = dataList.get(index);
            for(int j=index;j<endIndex;j++){
                //Take the next data item as label
                nextData = dataList.get(j+1);
                //create the features
                input.putScalar(new int[]{i, 0, j-index}, (double)curData/maxNum[0]);
                //create the label(s)
                label.putScalar(new int[]{i, 0, j-index}, (double)nextData/maxNum[0]);
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
