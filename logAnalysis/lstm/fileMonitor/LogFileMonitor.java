package org.deeplearning4j.logAnalysis.lstm.fileMonitor;

import java.io.File;

import org.apache.commons.io.monitor.FileAlterationListener;
import org.apache.commons.io.monitor.FileAlterationMonitor;
import org.apache.commons.io.monitor.FileAlterationObserver;

public class LogFileMonitor {
	FileAlterationMonitor monitor = null;
	public LogFileMonitor(long interval) throws Exception {
		monitor = new FileAlterationMonitor(interval);
	}

	public void monitor(String path, FileAlterationListener listener) {
		FileAlterationObserver observer = new FileAlterationObserver(new File(path));
		monitor.addObserver(observer);
		observer.addListener(listener);
	}
	public void stop() throws Exception{
		monitor.stop();
	}
	public void start() throws Exception {
		monitor.start();
	}
	public static void main(String[] args) throws Exception {
		LogFileMonitor m = new LogFileMonitor(5000);
		m.monitor(".\\logAnalysis\\logLstm\\",new LogFileListener());
		m.start();
	}
}
