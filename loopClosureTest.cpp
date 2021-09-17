#include "loopClosureScanContext.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include "laserSimulation.h"
#include "odomSimulation.h"

#include "scanContext.h"

bool poseDiffLargerThan( Eigen::Vector3f &poseOld, Eigen::Vector3f &poseNew )
{

        if( ( ( poseNew.head<2>() - poseOld.head<2>() ).norm() ) > 0.4f ){
                return true;
        }

        float angleDiff = ( poseNew.z() - poseOld.z() );

        if( angleDiff > M_PI ){
                angleDiff -= M_PI * 2.0f;
        }
        else if( angleDiff < -M_PI ){
                angleDiff += M_PI * 2.0f;
        }

        if( ::abs( angleDiff ) > 0.9f ){
                return true;
        }

        return false;
}

void dispalyOdom( cv::Mat &image, Eigen::Vector3f &pose )
{
        cv::Point2d point( pose(0) * 10 + 600, pose(1) * 10 + 800 );
        cv::circle(image, point, 4, cv::Scalar(0, 0, 255), 1);

        cv::imshow( "odom", image );
}


int main()
{
	std::cout<<" ------------------- Loop Closure Test ------------------- "<<std::endl;

	cv::Mat image = cv::Mat::zeros( 1200, 1200, CV_8UC3 );
	
	// 
	//slam::LoopClosureBase *detect = new slam::ScanContextLoopClosure();
		
	// scan context
	slam::ScanContext<float, 20> scanContext;


	slam::simulation::Simulation simulation;
        slam::simulation::OdomSimulation odomSim;

	// open the simulation file
        std::string file_name = "floor3.txt";
        simulation.openSimulationFile( file_name );

        std::string odom_file_name = "odom_data.txt";
        odomSim.openSimulationFile( odom_file_name );
	
	// LaserScan instance & ScanContainer instance
        slam::sensor::LaserScan scan;
	scan.angle_min = -3.12414f;
	scan.angle_max = 3.14159f;
	scan.angle_increment = 0.0043542264f;
	scan.range_min = 0.25000000030f;
	scan.range_max = 25.0000000000f;
	

        Eigen::Vector3f poseOld = Eigen::Vector3f::Zero();
        Eigen::Vector3f poseNew = poseOld;

	std::vector<Eigen::Vector3f> keyPoses;

	int keyCount = 0;	

	while( !simulation.endOfFile() ){

                // read a frame of data
                simulation.readAFrameData( scan );
                std::cout<<"frame count: "<<simulation.getFrameCount()<<std::endl;
		odomSim.readAFrameData( poseNew );
                std::cout<<"pose: "<<std::endl<<poseNew<<std::endl;

		if( poseDiffLargerThan( poseOld, poseNew ) ){
                        std::cerr<<"------------------ UPDATE ----------------"<<std::endl;
			keyCount ++;
			std::cout<<"keyCount = "<<keyCount <<std::endl;
			keyPoses.push_back( poseNew );

			// detect the loop closure
			scanContext.makeAndSaveScancontextAndKeys( scan );

			std::pair<int, float> ret = scanContext.detectLoopClosureID();
			int matchedID = ret.first;
		
			if( matchedID != -1 ){
			// draw the connected relationship
				cv::line( image, cv::Point2f( poseNew(0) * 10 + 600, poseNew(1) * 10 + 800 ),
						 cv::Point2f( keyPoses[matchedID](0) * 10 + 600 , keyPoses[matchedID](1) * 10 + 800 ),
						 cv::Scalar( 0, 255, 0 ), 1 );
				Eigen::MatrixXf scanContext1 = scanContext.getScanContext( matchedID );
				std::cout<<"scanContext2 : "<<std::endl<<scanContext1<<std::endl;
				Eigen::MatrixXf scanContext2 = scanContext.getScanContext( scanContext.getScanContextsSize() - 1 );
				std::cout<<"scanContext2 : "<<std::endl<<scanContext2<<std::endl;

			}

			dispalyOdom( image, poseNew );
		
			poseOld= poseNew;
		}
		
		cv::waitKey( 60 );
	}
	
	simulation.closeSimulationFile();
	odomSim.closeSimulationFile();

	return 0;
}
