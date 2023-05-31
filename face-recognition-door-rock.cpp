#include <iostream>
// for delay function.
#include <chrono>
#include <string>
// for signal handling
#include <signal.h>

#include <JetsonGPIO.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <fstream>

#define servo 12

using namespace std;
using namespace GPIO;
using namespace cv;
using namespace dnn;

VideoCapture cap1(0);

void signalingHandler(int signo);

int open_count = 0;

int main()
{
	//입력 이미지, 결과 감지의 클래스 인덱스, 해당 신뢰값 집합, 경계 상자값 집합의 Mat 변수
    Mat labels, stats, centroids, box;
	//cap1의 이미지를 받아올 Mat 자료형 변수
    Mat image;
	
    char savefile[200];        // 이미지 파일  이름을 200자 이내로 제한하기 위한 char 변수 선언


    signal(SIGINT, signalingHandler); //시그널 핸들러 함수

    GPIO::setwarnings(false);
    GPIO::setmode(GPIO::BCM);
	//문 잠금장치를 헤제하기위해 서보모터를 사용 주기를 20ms로 맞추기 위함
    GPIO::setup(servo, GPIO::OUT);
    GPIO::PWM p(servo, 50);

    p.start(13);

    cout << "PWM running. Press CTRL+C to exit." << endl;

    string class_name;
    std::vector<std::string> classes;
    std::ifstream file("choi.names");  //Input file stream = 파일로 부터 클래스 이름을 가지고와서 프로그램에 입력할 수 있도록 함

    while (std::getline(file, line)) {
        classes.push_back(line);
    }

	//Net 클래스는 다양한 레이어로 구성된 네트워크 구조를 표현하고 네트워크에서 특정 입력에 대한 순방향 실행을 지원한다.
	//opencv에서 지원하는 딥러닝 프레임워크 중 다크넷도 포함되어있다. Darknet 모델 파일 에 저장된 네트워크 모델을 읽는다.
	//순방향 실행을 위한 이미 만들어진 네트워크 Net 객체를 반환
    Net net = readNetFromDarknet("yolov4-tiny-custom.cfg", "weights/yolov4-tiny-custom_final.weights");

    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);

    while (1)
    {
        while (cap1.isOpened()) //웹캠이 정상적으로 작동될 시에
        {
			//카메라의 입력값을 받는다.
            cap1 >> image;
            bool isSuccess = cap1.read(image);
            if(!isSuccess)
            {
                cout << "Could not load the image!" << endl;
                break;
            }
			
			//영상의 프레임을 측정하기위한 시작 함수
            auto start = getTickCount();
               
			//model이란 변수에 기존 net 네트워크를 주입			   
            DetectionModel model = DetectionModel(net);
			//프레임에 대한 전처리 매개변수를 설정
			/*
			크기	새 입력 크기.
			평균	채널에서 빼는 평균값이 포함된 스칼라.
			규모	프레임 값의 승수입니다.
			스왑	RB 첫 번째 채널과 마지막 채널을 교환함을 나타내는 플래그입니다.
			수확	크기 조정 후 이미지를 자를지 여부를 나타내는 플래그입니다. blob(n, c, y, x) = 크기 조정 * 크기 조정( frame(y, x, c) ) - 평균(c) )
			*/
            model.setInputParams(1 / 255.0, Size(416, 416), Scalar(), true);
                
            std::vector<int> classIds; //학습 시킬때 분류한 클래스 변수
            std::vector<float> scores; //박스에 들어가있는 이미지가 해당 클래스인 확률 변수
            std::vector<Rect> boxes; //경계 상자값 집합 변수
            model.detect(image, classIds, scores, boxes, 0.98, 0.02);
            //detect(입력 이미지, 결과 감지의 클래스 인덱스, 해당 신뢰값 집합, 경계 상자값 집합, 임계값, 비임계값)
                
			//영상의 프레임을 측정하기위한 끝내기 함수	
            auto end = getTickCount();

            for (int i = 0; i < classIds.size(); i++) 
            {
                //여기는 그냥 객체가 몇 개에 따라 classIds.size가 달라짐
                //ex)아무 것도 없으면 for문 동작x, 객체 하나: 0, 객체 둘: 1... 객체 n: n-1
                rectangle(image, boxes[i], Scalar(0, 255, 0), 2);
                    
                char text[100];
                snprintf(text, sizeof(text), "%s: %.2f", classes[classIds[i]].c_str(), scores[i]);
                //0번째 객체에 해당 객체의 클래스 정보와 신뢰값을 대입, 1번째 객체에 해당 ~~
                putText(image, text, Point(boxes[i].x, boxes[i].y - 5), cv::FONT_HERSHEY_SIMPLEX, 1,
                Scalar(0, 255, 0), 2);
                    
                if(classIds[i] <= 0) //만약 카메라 입력 객체가 최재영일 경우
                {
                    if((open_count++) >= 10)
                    {
                        p.ChangeDutyCycle(7); //문 잠금 헤제
                        sleep(11);
                        p.ChangeDutyCycle(13); //문 잠금
                        sleep(2);
                        open_count = 0;
                    }
                }

                else if(classIds[i] == 1) //만약 카메라 입력 객체가 최재영이 아닐 경우
                {
                    p.ChangeDutyCycle(13); //문 잠금
                }

                else if(classIds[i] == 2) // 만약 카메라 입력 객체가 모자 쓴 최재영일 경우
                {
                    if((open_count++) >= 10)
                    {
                        p.ChangeDutyCycle(7); //문 잠금 헤제
                        sleep(11);
                        p.ChangeDutyCycle(13); //문 잠금
                        sleep(2);
                        open_count = 0;
                    }
                }

                else
                {  
                    if((open_count++) >= 10)
                    {
                        p.ChangeDutyCycle(7);
                        sleep(11);
                        p.ChangeDutyCycle(13);
                        sleep(2);
                        open_count = 0;
                    }
                    p.ChangeDutyCycle(13);
                    /*sprintf(savefile, "image.jpg");
                    imwrite(savefile, image);     // img를 파일로 저장한다.*/
                }
            }
            auto totalTime = (end - start) / getTickFrequency();
             // 영상 프레임 화면 왼쪽 상단에 출력
            putText(image, "FPS" + to_string(int(1 / totalTime)), Point(50,50), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 2, false);
            imshow("Image", image);

            int k = waitKey(10);
            if (k == 113)break;
        }

    }
    p.stop();
    return 0;
}

void signalingHandler(int signo) //마지막 처리를 위한 signal 핸들러, 함수 모두 OFF
{
    printf("\n\nGoodbye World\n\n");
    cap1.release();
    GPIO::cleanup();
    exit(signo);
}
