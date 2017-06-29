package org.deeplearning4j.logAnalysis.lstm;

public class LogBean {
private long time;
private int hostID;
private int programID;
private int severity;

public long getTime() {
	return time;
}
public void setTime(long time) {
	this.time = time;
}
public int getHostID() {
	return hostID;
}
public void setHostID(int hostID) {
	this.hostID = hostID;
}
public int getProgramID() {
	return programID;
}
public void setProgramID(int programID) {
	this.programID = programID;
}
public int getSeverity() {
	return severity;
}
public void setSeverity(int severity) {
	this.severity = severity;
}

@Override
public String toString()
{
	StringBuilder sb=new StringBuilder();
	sb.append("Time="+this.time+", ");
	sb.append("Host ID=="+this.hostID+", ");
	sb.append("Program ID="+this.programID+", ");
	sb.append("Severity="+this.severity);
	return sb.toString();
}
}