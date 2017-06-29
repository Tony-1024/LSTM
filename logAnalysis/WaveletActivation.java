package org.deeplearning4j.logAnalysis;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.factory.Nd4j;
import jwave.Transform;
import jwave.TransformBuilder;
import jwave.transforms.wavelets.Wavelet;

public class WaveletActivation extends BaseActivationFunction {
	String waveletIdentifier;
	public WaveletActivation(String waveletIdentifier)
	{
		super();
		this.waveletIdentifier=waveletIdentifier;
	}
	
	@Override
	public INDArray getActivation(INDArray in, boolean training) {
        double[][] arr=new double[in.rows()][in.columns()];
        for(int i=0; i<in.rows(); i++)
        	for(int j=0; j<in.columns(); j++)
        		arr[i][j]=in.getDouble(i, j);
        String transformIdentifier="Wavelet Packet Transform";
        
        Transform transform = TransformBuilder.create( transformIdentifier, waveletIdentifier );
        double[][] arrFreqOrHilb = transform.forward( arr );
        
        INDArray arrTransfered = Nd4j.create(arrFreqOrHilb);
        
        Nd4j.getExecutioner().execAndReturn(new Tanh(arrTransfered));
		return arrTransfered;
	}

	@Override
	public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        double[][] arr=new double[in.rows()][in.columns()];
        for(int i=0; i<in.rows(); i++)
        	for(int j=0; j<in.columns(); j++)
        		arr[i][j]=in.getDouble(i, j);
        String transformIdentifier="Wavelet Packet Transform";
        Transform transform = TransformBuilder.create( transformIdentifier, waveletIdentifier );
        double[][] arrFreqOrHilb = transform.forward( arr );
        INDArray arrTransfered = Nd4j.create(arrFreqOrHilb);
        
        INDArray dLdz = Nd4j.getExecutioner().execAndReturn(new Tanh(arrTransfered).derivative());
        //Multiply with epsilon
        dLdz.muli(epsilon);
        return new Pair<>(dLdz, null);
	}

}
