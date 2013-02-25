#include "PHD.h"
#include <iostream>
#include <fstream>

void testCholesky () {
  cv::Matx<double, 5, 5> A1 = cv::Matx<double, 5, 5>::eye();
  cv::Matx<double, 5, 5> A2;
  A2 <<  1 , -1 , -1 , -1 , -1,
        -1 ,  2 ,  0 ,  0 ,  0,
        -1 ,  0 ,  3 ,  1 ,  1,
        -1 ,  0 ,  1 ,  4 ,  2,
        -1 ,  0 ,  1 ,  2 ,  5 ;
  cv::Matx<double, 8, 8> A3;
  A3 << 1, 1, 1, 1, 1, 1, 1, 1,
        1, 2, 3, 4, 5, 6, 7, 8,
        1, 3, 6, 10, 15, 21, 28, 36,
        1, 4, 10, 20, 35, 56, 84, 120,
        1, 5, 15, 35, 70, 126, 210, 330,
        1, 6, 21, 56, 126, 252, 462, 792,
        1, 7, 28, 84, 210, 462, 924, 1716,
        1, 8, 36, 120, 330, 792, 1716, 3432;
  cv::Matx<double, 5, 5> C1, C2;
  cv::Matx<double, 8, 8> C3;
  bool B1 = cholesky<5>(A1, C1);
  bool B2 = cholesky<5>(A2, C2);
  bool B3 = cholesky<8>(A3, C3);
  std::cout << A1 << std::endl << C1 << std::endl;
  std::cout << A2 << std::endl << C2 << std::endl;
  std::cout << A3 << std::endl << C3 << std::endl;
}

void testRandn() {
  cv::Vec<double, 4> mean; // = cv::Vec<double, 4>::zeros(); // FIXME
  mean << 0, 0, 0, 0; // Why does the above not work?
  cv::Matx<double, 4, 4> cov;
  cov << 2, 0.2, 0, 0,
       0.2, 1, 0, 0,
       0, 0, 0.7, -0.2,
       0, 0, -0.2, 1;
  std::vector<cv::Vec<double, 4> > elems =
    sampleMVGaussian<4> (mean, cov, 1000);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 1000; ++j) {
      std::cout << elems[j](i);
      if (j != 999) {std::cout << ",";}
    }
    std::cout << "NEWLINE"; // Why doesn't "\n" work?
  }
}

void testMerge() {
  cv::Vec<double, 2> C1, C2;
  C1 << 0, 0;
  C2 << 10, 10;
  cv::Matx<double, 2, 2> P = 0.5*cv::Matx<double, 2, 2>::eye();
  std::vector<WeightedGaussian<2> > elements;
  WeightedGaussian<2> W1;
  WeightedGaussian<2> W2; 
  std::vector<cv::Vec<double, 2> > M1, M2;
  cv::Matx<double, 2, 2> P1, P2;
  for (int i = 0; i < 20; ++i) {
    // Create means and covariances
    M1 = sampleMVGaussian(C1, P, 1);
    M2 = sampleMVGaussian(C2, P, 1);
    P1 = cv::Matx<double, 2, 2>::zeros();
    P1 << 0.5 + cv::theRNG().gaussian(0.1), 0, 0.1+cv::theRNG().gaussian(0.1),
       0.6 + cv::theRNG().gaussian(0.05);
    P2 = cv::Matx<double, 2, 2>::zeros();
    P2 << 0.4 + cv::theRNG().gaussian(0.04), 0, 0.2+cv::theRNG().gaussian(0.1),
       0.7 + cv::theRNG().gaussian(0.05);
    P1 += P1.t();
    P2 += P2.t();
    W1 = WeightedGaussian<2>(1., M1[0], P1);
    W2 = WeightedGaussian<2>(1., M2[0], P2);
    elements.push_back(W1);
    elements.push_back(W2);
  }
  GaussianMixture<2> mixture(elements);
  std::ofstream out;
  out.open("bmerge.txt");
  for (int i = 0; i < mixture.size(); ++i) {
    out << mixture.at(i).getWeight() << ",";
    out << mixture.at(i).getMean()(0) << ",";
    out << mixture.at(i).getMean()(1) << ",";
    out << mixture.at(i).getCov()(0, 0) << ",";
    out << mixture.at(i).getCov()(0, 1) << ",";
    out << mixture.at(i).getCov()(1, 0) << ",";
    out << mixture.at(i).getCov()(1, 1) << "\n";
  }
  out.close();
  mixture.merge(2);
  out.open("amerge.txt")
  for (int i = 0; i < mixture.size(); ++i) {
    out << mixture.at(i).getWeight() << ",";
    out << mixture.at(i).getMean()(0) << ",";
    out << mixture.at(i).getMean()(1) << ",";
    out << mixture.at(i).getCov()(0, 0) << ",";
    out << mixture.at(i).getCov()(0, 1) << ",";
    out << mixture.at(i).getCov()(1, 0) << ",";
    out << mixture.at(i).getCov()(1, 1) << "\n";
  }
}

int main() {
  testMerge();
  return 0;
}
