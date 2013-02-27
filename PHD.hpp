#include <cmath>
#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"

/* Classes */

/* Basic class to represent a weighted vector-valued Gaussian function which
 * is concentrated around a mean and scattered according to a covariance
 * matrix. */

template <int D>
class WeightedGaussian {
  protected:
    double mWeight;
    cv::Vec<double, D> mMean;
    cv::Matx<double, D, D> mCov;
    int mDim;
  public:
    WeightedGaussian<D>();
    WeightedGaussian<D>(double weight, cv::Vec<double, D> & mean,
        cv::Matx<double, D, D> & covariance);
    // Copy constructor
    WeightedGaussian<D>(const WeightedGaussian<D> & src);
    ~WeightedGaussian<D>();
    // Accessors
    double getWeight() const {return mWeight;};
    cv::Vec<double, D> getMean() const {return mMean;};
    cv::Matx<double, D, D> getCov() const {return mCov;};
    int getDim() const {return mDim;};
    void setWeight (double & weight) {mWeight = weight;};
    void setMean (cv::Vec<double, D> & mean) {mMean = mean;};
    void setCov (cv::Matx<double, D, D> & cov) {mCov = cov;};
    // Operators
    // Assignment operator
    WeightedGaussian<D> & operator= (const WeightedGaussian<D> & src);
    // Comparison operators compare weights:
    bool operator<(WeightedGaussian cmp) const {
      return mWeight < cmp.getWeight();};
    bool operator<=(WeightedGaussian cmp) const {
      return mWeight <= cmp.getWeight();};
    bool operator>(WeightedGaussian cmp) const {
      return mWeight > cmp.getWeight();};
    bool operator>=(WeightedGaussian cmp) const {
      return mWeight >= cmp.getWeight();};
    // Methods
    double evaluate (cv::Vec<double, D> argument);
};

template <int D>
class GaussianMixture {
  public:
    std::vector<WeightedGaussian<D> > mComponents;
    GaussianMixture();
    GaussianMixture(std::vector<WeightedGaussian<D> > components);
    // Accessors
    WeightedGaussian<D> & at (int i) {return mComponents.at(i);};
    int size () { return mComponents.size();}
    void add (WeightedGaussian <D> gm) {mComponents.push_back(gm);};
    // Methods
    double evaluate(cv::Vec<double, D> argument);
    void merge (double theshold);
    void trim (double threshold);
    void truncate (unsigned int threshold);
};

/* Abstract motion and measurement model classes to implement as whichever
   model is appropriate, with the required vector dimensions. */

template <int D>
class MotionModel {
  protected:
    bool mIsLinear;
    cv::Matx<double, D, D> mJacobian;
    cv::Matx<double, D, D> mProcNoise;
  public:
    cv::Vec<double, D> predict(cv::Vec<double, D> x) = 0;
    // Accessors
    bool isLinear() const {return mIsLinear;};
    cv::Matx<double, D, D> getJacobian() const {return mJacobian;};
    cv::Matx<double, D, D> getProcNoise() const {return mProcNoise;};
    void setIsLinear(bool isLinear) {mIsLinear = isLinear;};
    void setJacobian(cv::Matx<double, D, D> jacobian) {mJacobian = jacobian;};
    void setProcNoise(cv::Matx<double, D, D> procNoise) {
      mProcNoise = procNoise;};
    // Operators
    cv::Vec<double, D> operator()(cv::Vec<double, D> x) {return predict(x);};
};

template <int D, int M>
class MeasurementModel {
  protected:
    bool mIsLinear;
    cv::Matx<double, D, M> mJacobian;
    cv::Matx<double, M, M> mMeasNoise;
  public:
    cv::Vec<double, M> predict(cv::Vec<double, D> x) = 0;
    // Accessors
    bool isLinear() const {return mIsLinear;};
    cv::Matx<double, D, M> getJacobian() const {return mJacobian;};
    cv::Matx<double, M, M> getMeasNoise() const {return mMeasNoise;};
    void setIsLinear(bool isLinear) {mIsLinear = isLinear;};
    void setJacobian(cv::Matx<double, D, D> jacobian) {mJacobian = jacobian;};
    void setMeasNoise(cv::Matx<double, M, M> measNoise) {
      mMeasNoise = measNoise;};
    // Operators
    cv::Vec<double, M> operator()(cv::Vec<double, D> x) {return predict(x);};
};

template <int D>
class ConstantPositionMotionModel: public MotionModel {
  public:
    ConstantPositionMotionModel(cv::Matx<double, D, D> procNoise);
    cv::Vec<double, D> predict(cv::Vec<double, D> x) {return x;};
}

template <int D, int M>
class GMPHDFilter {
  protected:
    MotionModel * mMotionModel;
    MeasurementModel * mMeasurementModel;
    GaussianMixture<D> mPHD;
    void predictLinear();
    void predictNonLinear();
    void updateLinear(std::vector<cv::Vec<double, M> > measurements);
    void updateNonLinear(std::vector<cv::Vec<double, M> > measurements);
  public:
    // FIXME Default constructor? PHDFilter();
    PHDFilter(MotionModel * motionModel, MeasurementModel * measurementModel,
        double ps = 0.99, double pd = 0.9, double k = 3e-3,
        double mt = 1., double trit = 1e-3, int trut = 120);
    // Accessors
    GaussianMixture<d> getPHD() {return mPHD;};
    MotionModel * getMotionModel() {return mMotionModel;};
    MotionModel * getMeasurementModel() {return mMeasurementModel;};
    // Methods
    void predict();
    void update(std::vector<cv::Vec<double, M> >);
    std::vector<cv::Vec<double, D> > getStateEstimate();
    // Parameters
    double mProbSurvival;
    double mProbDetection;
    double mClutterDensity;
    double mMergeThreshold;
    double mTrimThreshold;
    int mTruncThreshold;
};

/* Functions */

/* Cholesky decomposition of symmetric and positive definite matrix. Can also
 * be used to efficiently verify the positive-definiteness of input matrix by
 * its boolean return value. Input matrix A is first argument, and output L is
 * stored in second argument such that L*L' = A. Note that although positive
 * definiteness is checked for, symmetry is not verified. If the input matrix
 * is nonsymmetric, the output is not valid. */
template <int D>
bool cholesky (const cv::Matx<double, D, D> & mat,
    cv::Matx<double, D, D> & output) {
  double sum;
  output = cv::Matx<double, D, D>::zeros();
  for (int i = 0; i < D; ++i) {
    for (int j = 0; j <= i; ++j) {
      sum = mat(i, j);
      for (int k = 0; k < j ; ++k) {
        sum -= output(i, k) * output(j, k);
      }
      if (i == j) {
        if (sum <= 0) {
          return false; // Matrix is not positive definite
        }
        output(i, j) = sqrt(sum);
      } else {
        output(i, j) = sum/output(j, j);
      }
    }
  }
  return true; // Matrix is positive definite
}

/* Evaluates the probability distribution function (pdf) of a Multivariate
 * Normal random variable evaluated at input vector X. The parameters of the
 * distribution are its mean (Nx1 vector) and its covariance (NxN matrix).*/
template <int D>
double MVNormalPDF (cv::Vec<double, D> mean,
    cv::Matx<double, D, D> cov,
    cv::Vec<double, D> x) {
  double d = double(D);
  double val = pow(2*M_PI, -d/2) * pow(cv::determinant(cov), -0.5) *
    exp( -0.5 * (x - mean).t() * cov.inv() * (x - mean) );
  return val;
}

template <int D>
std::vector<cv::Vec<double, D> > sampleMVGaussian (
    const cv::Vec<double, D> & mean,
    const cv::Matx<double, D, D> & cov,
    int n) {
  cv::Matx<double, D, D> dCov;
  bool decSuccess = cholesky<D>(cov, dCov);
  if (decSuccess == false) {
    std::cout << "Failed to decompose covariance matrix\n";
    exit(1);
  }
  cv::Vec<double, D> randVector;
  cv::RNG rng = cv::theRNG();
  std::vector<cv::Vec<double, D> > samples;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < D; ++j) {
      randVector(j) = rng.gaussian(1.0);
    }
    samples.push_back(dCov * randVector + mean);
  }
  return samples;
}

/* Implementation */

template <int D>
WeightedGaussian<D> :: WeightedGaussian () {
  mWeight = 1.0;
  for (int i = 0; i < D; ++i) {mMean(i) = 0;}
  mCov = cv::Matx<double, D, D>::eye();
  mDim = D;
}

template <int D>
WeightedGaussian<D> :: WeightedGaussian (
    double weight,
    cv::Vec<double, D> & mean,
    cv::Matx<double, D, D> & covariance) {
  mWeight = weight;
  mMean = mean;
  mCov = covariance;
  mDim = D;
}

template <int D>
WeightedGaussian<D> :: WeightedGaussian (const WeightedGaussian<D> & src) {
  mWeight = src.mWeight;
  mMean = src.mMean;
  mCov = src.mCov;
  mDim = D;
}

template <int D>
WeightedGaussian<D> :: ~WeightedGaussian() {
};

template <int D>
WeightedGaussian<D> & WeightedGaussian<D> :: operator= (
    const WeightedGaussian<D> & src) {
  if (this != &src) {
    mWeight = src.getWeight();
    mMean = src.getMean();
    mCov = src.getCov();
    mDim = D;
  }
  return *this;
}
    
template <int D>
double WeightedGaussian<D> :: evaluate (cv::Vec<double, D> x) {
  double val = mWeight * MVNormalPDF<D> (mMean, mCov, x);
  return val;
}

/* The following is probably not required:
template <int D>
GaussianMixture<D> :: GaussianMixture () {
  std::vector<WeightedGaussian<D> > mComponents;
}
*/

template <int D>
GaussianMixture<D> :: GaussianMixture (
    std::vector<WeightedGaussian<D> > components) {
  mComponents = components;
}

template <int D>
double GaussianMixture<D> :: evaluate (cv::Vec<double, D> x) {
  double val = 0;
  typename std::vector<WeightedGaussian<D> >::iterator it;
  for (it = mComponents.begin(); it != mComponents.end(); ++it) {
    val += it->evaluate(x);
  }
  return val;
}
/* Merge Gaussian components that are close together in the sense of the
 * Mahalanobis distance */
template <int D>
void GaussianMixture<D> :: merge (double threshold) {
  // Sort the Gaussian Mixture in descending order according to weight
  std::sort(mComponents.begin(), mComponents.end(),
      std::greater<WeightedGaussian<D> >());
  // Create a vector to hold the merged Gaussian Mixture
  std::vector<WeightedGaussian<D> > mergedMixture;
  // Use typename so the compiler doesn't get confused about containers of
  // templated classes
  typename std::vector<WeightedGaussian<D> >::iterator it;
  typename std::vector<WeightedGaussian<D> >::iterator jt;
  typename std::vector<WeightedGaussian<D> >::iterator kt;
  // Auxiliary vector to store the components to merge
  std::vector<WeightedGaussian<D> > toMerge;
  double mergedWeight;
  cv::Vec<double, D> mergedMean;
  cv::Matx<double, D, D> mergedCov;
  double distance;
  while (!mComponents.empty()) {
    toMerge.clear();
    it = mComponents.begin();
    mergedWeight = it->getWeight();
    jt = it + 1;
    // Compare each element in the mixture with the element with the greatest
    // weight. Set aside the elements with Mahalanobis distance less than the
    // threshold in vector toMerge.
    while (jt != mComponents.end()) {
      distance = cv::Mahalanobis(it->getMean(), jt->getMean(),
          jt->getCov().inv());
      if (distance < threshold) {
        toMerge.push_back(*jt);
        mergedWeight += jt->getWeight();
        jt = mComponents.erase(jt);
      } else {
        ++jt;
      }
    }
    if (toMerge.empty()) {
      mergedMixture.push_back(*it);
      it = mComponents.erase(it);
    } else {
      // Compute the mean
      mergedMean = (it->getWeight()) * (it->getMean());
      for (kt = toMerge.begin(); kt != toMerge.end(); ++kt) {
        mergedMean += (kt->getWeight()) * (kt->getMean());
      }
      mergedMean = (1/mergedWeight) * mergedMean;
      // Compute the covariance
      mergedCov = it->getWeight() * (it->getCov() +
          (mergedMean - it->getMean()) * (mergedMean - it->getMean()).t());
      for (kt = toMerge.begin(); kt != toMerge.end(); ++kt) {
        mergedCov = kt->getWeight() * (kt->getCov() +
            (mergedMean - kt->getMean()) * (mergedMean - kt->getMean()).t());
      }
      mergedCov = (1/mergedWeight) * mergedCov;
      // Add merged component to merged mixture
      mergedMixture.push_back(WeightedGaussian<D>(mergedWeight, mergedMean,
          mergedCov));
      // Remove the merged element
      it = mComponents.erase(it);
    }
  }
  // Update the mixture
  mComponents = mergedMixture;
}

/* Deletes components of the Gaussian mixture with weights under a given
 * threshold */
template <int D>
void GaussianMixture<D> :: trim (double threshold) {
  // Can't wait for the auto keyword
  typename std::vector<WeightedGaussian<D> >::iterator it = 
    mComponents.begin();
  while (it != mComponents.end()) {
    if (it->getWeight() <= threshold) {
      it = mComponents.erase(it);
    } else {
      ++it;
    }
  }
}

/* If the Gaussian Mixture has more elements than are supported, truncate to the desired number of components */
template <int D>
void GaussianMixture<D> :: truncate (unsigned int threshold) {
  unsigned int size = mComponents.size();
  if (size > threshold) {
    // Sort the Gaussian Mixture in descending order according to weight
    std::sort(mComponents.begin(), mComponents.end(),
        std::greater<WeightedGaussian<D> >());
    mComponents.erase(mComponents.begin() + threshold + 1, mComponents.end());
  }
}

template <int D>
ConstantPositionMotionModel<D> :: ConstantPositionMotionModel(
    cv::Matx<double, D, D> procNoise) {
  mIsLinear = true;
  mJacobian = cv::Matx<double, D, D>::eye();
  mProcNoise = procNoise;
}

template <int D, int M>
void GMPHDFilter<D, M> :: predictLinear() {
  typename std::vector<WeightedGaussian<D> >::iterator it;
  cv::Matx<double, D, D> J = mMotionModel->getJacobian();
  cv::Matx<double, D, D> Q = mMotionModel->getProcNoise();
  for (it = mPHD.mComponents.begin(); it != mComponents.end(); ++it) {
    // Scale weight according to probability of survival
    it->setWeight(mProbSurvival * it->getWeight());
    // Use the predict method of the motion model
    it->setMean( mMotionModel -> predict( it->getMean() ) );
    // Use Jacobian to predict covariance
    it->setCov( J * (it->getCov()) * J.t() + Q);
  }
}

template <int D, int M>
void GMPHDFilter<D, M> :: predictNonLinear() {
  // UK PHD Prediction
  // TODO Implement
}

template <int D, int M>
void GMPHDFilter<D, M> :: updateLinear(
    std::vector<cv::Vec<double, M> > measurements) {
  // Copy of prior
  std::vector<WeightedGaussian<D> > prior(mPHD.mComponents);
  // Components for updated terms
  std::vector<cv::Vec<double, M> > eta;
  std::vector<cv::Matx<double, M, M> S;
  std::vector<cv::Matx<double, D, M> K;
  std::vector<cv::Matx<double, D, D> P;
  R = mMeasurementModel.getMeasNoise();
  H = mMeasurementModel.getJacobian();
  ID = cv::Matx<double, D, D>::eye();
  typename std::vector<WeightedGaussian<D> >::iterator it;
  typename std::vector<cv::Vec<double, D> >::iterator jt;
  // Update previous weights and compute elements for update
  for (it = mPHD.mComponents.begin(); it != mComponents.end(); ++it) {
    eta.push_back(H * it->getMean());
    S.push_back(R + H * (it->getCov()) * H.t());
    K.push_back((it->getCov()) * H.t() * S.back().inv());
    P.push_back((I - K.back() * H)*(it->getCov()));
    // Scale weight according to probability of detection
    it->setWeight((1 - mProbDetection) * it->getWeight());
  }
  // Generate new Gaussian terms for each one of the measurements
  double nWeight;
  cv::Vec<double, D> nMean;
  for (jt = measurements.begin(); jt != measurements.end(); ++jt) {
    for (int i = 0; i < prior.size(); ++i) {
      nWeight = mProbDetection * (it->getWeight()) * MVNormalPDF<M>(
          eta[i], S[i], *jt );
      nMean = prior[i].getMean() + K[i]*(*jt - eta[i]);
      mPHD.mComponents.push_back(WeightedGaussian<D>(nWeight, nMean, P[i]));
    }
  }
  //TODO Finish this
}







template <int D, int M>
GMPHDFilter<D, M> :: GMPHDFilter(MotionModel * motionModel,
    MeasurementModel * measurementModel, double ps, double pd,
    double k, double mt, double trit, int trut) {
  mMotionModel = motionModel;
  mMeasurementModel = measurementModel;
  mProbSurvival = ps;
  mProbDetection = pd;
  mClutterDensity = k;
  mMergeThreshold = mt;
  mTrimThreshold = trit;
  mTruncThreshold = trut;
}

template <int D, int M>
void GMPHDFilter<D, M> :: predict() {
  if (mMotionModel->isLinear()) {
    predictLinear();
  } else {
    predictNonLinear();
  }
}

template <int D, int M>
void GMPHDFilter<D, M> :: update(
    std::vector<cv::Vec<double, M> > measurements) {
  if (mMeasurementModel->isLinear()) {
    updateLinear(std::vector<cv::Vec<double, M> > measurements);
  } else {
    updateNonLinear(std::vector<cv::Vec<double, M> > measurements);
  }
}