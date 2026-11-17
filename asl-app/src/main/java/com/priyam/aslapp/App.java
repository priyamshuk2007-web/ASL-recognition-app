package com.priyam.aslapp;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;
import nu.pattern.OpenCV;

public class App{
    public static void main(String[] args)
    {
        OpenCV.loadLocally();
        VideoCapture camera = new VideoCapture(0);
        if(!camera.isOpened())
        {
            System.out.println("Could not open camera");
            return;
        }
        Mat frame = new Mat();
        while(true)
        {
            camera.read(frame);
            if(frame.empty())
            {
                continue;
            }
            HighGui.imshow("ASL App - Camera Feed", frame);
            int key = HighGui.waitKey(1);
            if(key != -1)
            {
                System.out.println("Key pressed: " +key);
            }
            if(key=='Q')
            {
                break;
            }
        }
        camera.release();
        HighGui.destroyAllWindows();
    }
}
