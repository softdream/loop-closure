#include "icp.h"
#include <cmath>
#include <iostream>

namespace slam{

ICP::ICP()
{

}

ICP::~ICP()
{

}

const float ICP::solveICP( ScanContainer& A_src, ScanContainer& B_src )
{
	A.clear();
	std::vector<Eigen::Vector2f> B; 
	std::vector<Eigen::Vector2f> B_apostrophe( B_src.getSize(), Eigen::Vector2f( 0.0f, 0.0f ) );	

	for( size_t i = 0; i < A_src.getSize(); i ++ ){
		A.push_back( A_src.getIndexData(i) );
	}	
	
	for( size_t i = 0; i < B_src.getSize(); i ++ ){
                B.push_back( B_src.getIndexData(i) );
        }

	// 1. Caculate the center of point cloud of A
	// A_center = ( A_1 + A_2 + ...A_n ) / n;
	Eigen::Vector2f Asum(0.0f, 0.0f);
	for( auto it : A ){
		Asum += it;
	}
	Acenter = Asum / A.size();

	// 2. Caculate the distance from points to the center of A
	// a_i = A_i - A_center, i = 1, 2, ...., n
	a.clear();
	for( auto it : A ){
		Eigen::Vector2f tmp = it - Acenter;
		a.push_back( tmp );
	}
	
	// begin iterate
	int iteration = 0;
	float loss = 0.0f;
	
	// Error Function : 
	// E(R, T) = Sigma(i = 1 to n){a_i * R * b_i} / n

	// E(R, T) = Sigma(i = 1 to n){ cos(theta) * (a_i(x) * b_i(x) + a_i(y) * b_i(y)) + sin(theta) * (a_i(x) * b_i(y) - a_i(y) * b_i(x)) } / n

	// dE(R, T) / d(theta) = Sigma(i = 1 to n){-sin(theta) * (a_i(x) * b_i(x) + a_i(y) * b_i(y)) + cos(theta) * ( a_i(x) * b_i(y) - a_i(y) * b_i(x) ) } / n

	// let dE(R, T) / d(theta) = 0

	// tan(theta) = Sigma(i = 1 to n){( a_i(x) * b_i(y) - a_i(y) * b_i(x) )} / Sigma(i = 1 to n){ (a_i(x) * b_i(x) + a_i(y) * b_i(y)) }
	while( iteration < maxIteration ){
		loss = iterateOnce( B, B_apostrophe );
		
		B.swap( B_apostrophe );	
		iteration ++;
	}

	return loss;
}

float ICP::iterateOnce( std::vector<Eigen::Vector2f>& B, std::vector<Eigen::Vector2f>& B_apostrophe )
{
	Eigen::Vector2f Bsum(0.0f, 0.0f);

	for( auto it : B ){
		Bsum += it;
	}
	Bcenter = Bsum / B.size();

	b.clear();
	for( auto it : B ){
		Eigen::Vector2f tmp = it - Bcenter;
		b.push_back( tmp );
	}

	// 3. caculate the rotate theta
	// theta = arctan( Sigma(i = 1 to n){( a_i(x) * b_i(y) - a_i(y) * b_i(x) )} / Sigma(i = 1 to n){ (a_i(x) * b_i(x) + a_i(y) * b_i(y)) } );
	float y = 0.0f, x = 0.0f;
	for( size_t i = 0; i < b.size(); i ++ ){
		y += ( a[i](0) * b[i](1) ) - ( a[i](1) * b[i](0) );
	
		x += ( a[i](0) * b[i](0) ) + ( a[i](1) * b[i](1) );
	}
	float theta = ::atan2( y, x );
	std::cout<<"theta = "<<theta<<std::endl;	

	// 4. get the rotate matrix and transfrom matrix
	//	R = |cos(theta), -sin(theta)|
	//	    |sin(theta),  cos(theta)|
	R( 0, 0 ) = ::cos(theta);
	R( 0, 1 ) = -::sin(theta);
	R( 1, 0 ) = ::sin(theta);
	R( 1, 1 ) = ::cos(theta);
	
	// T = A_center - R * B_center
	T = Acenter - R * Bcenter;

	// 5. get new pose B' from B
	// B_apostrophe_i = R * B_i + T
	for( size_t i = 0; i < B.size(); i ++ ){
		B_apostrophe[i] = R * B[i] + T;
	}

	// 6. caculate loss function
	// Loss = Sigma(i = 1 to n){(A_i - B_apostrophe_i) * (A_i - B_apostrophe_i)} / n
	float loss = 0.0f;
	for( size_t i = 0; i < B.size(); i ++ ){
		Eigen::Vector2f tmp = A[i] - B[i];
		loss += tmp.squaredNorm();
	}
	loss /= B.size();

	return loss;
}


}






