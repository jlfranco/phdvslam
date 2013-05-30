#include <cmath>
#include <vector>
#include <iostream>
#include <cassert>
#include "opencv2/core/core.hpp"

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
    void setWeight (double weight) {mWeight = weight;};
    void setMean (cv::Vec<double, D> mean) {mMean = mean;};
    void setCov (cv::Matx<double, D, D> cov) {mCov = cov;};
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
    GaussianMixture<D>();
    GaussianMixture<D>(std::vector<WeightedGaussian<D> > components);
    // Accessors
    WeightedGaussian<D> & at (int i) {return mComponents.at(i);};
    int size () { return mComponents.size(); }
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
    MotionModel<D> ();
    MotionModel<D> (const MotionModel<D> & src);
    virtual ~MotionModel<D>(){};
    virtual MotionModel<D> * copy() const = 0;
    virtual cv::Vec<double, D> predict(cv::Vec<double, D> x) = 0;
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
    cv::Matx<double, M, D> mJacobian;
    cv::Matx<double, M, M> mMeasNoise;
  public:
    MeasurementModel<D, M> ();
    MeasurementModel<D, M> (const MeasurementModel<D, M> & src);
    virtual ~MeasurementModel<D, M>(){};
    virtual MeasurementModel<D, M> * copy() const = 0;
    virtual cv::Vec<double, M> predict(cv::Vec<double, D> x) = 0;
    virtual WeightedGaussian<D> invert(cv::Vec<double, M> z) = 0;
    // Accessors
    bool isLinear() const {return mIsLinear;};
    cv::Matx<double, M, D> getJacobian() const {return mJacobian;};
    cv::Matx<double, M, M> getMeasNoise() const {return mMeasNoise;};
    void setIsLinear(bool isLinear) {mIsLinear = isLinear;};
    void setJacobian(cv::Matx<double, M, D> jacobian) {mJacobian = jacobian;};
    void setMeasNoise(cv::Matx<double, M, M> measNoise) {
      mMeasNoise = measNoise;};
    // Operators
    cv::Vec<double, M> operator()(cv::Vec<double, D> x) {return predict(x);};
};

// Represents the class of motion models where the state dynamics are linear, so
// that the state at time k+1 can be obtained from multiplying a matrix by the
// previous state

template <int D>
class LinearMotionModel: public MotionModel<D> {
  public:
    // This constructor uses the given dynamics matrix and process noise to
    // build the motion model
    LinearMotionModel<D>(cv::Matx<double, D, D> dynamicsMatrix,
        cv::Matx<double, D, D> procNoise);
    ~LinearMotionModel<D>() {};
    MotionModel<D> * copy() const;
    //LinearMotionModel<D>(const LinearMotionModel<D> & src);
    // This constructor assumes that the dynamics matrix is the identity
    // (Constant position)
    LinearMotionModel(cv::Matx<double, D, D> procNoise);
    // Uses the dynamics matrix to predict the next state, given that the
    // current state is 'x'
    cv::Vec<double, D> predict(cv::Vec<double, D> x);
};

template <int D, int M>
class LinearMeasurementModel: public MeasurementModel<D, M> {
  protected:
    cv::Matx<double, D, M> mMeasMatrixPseudoInverse;
    cv::Matx<double, D, D> mNewComponentCov;
  public:
    LinearMeasurementModel<D, M> (cv::Matx<double, M, D> measurementMatrix,
        cv::Matx<double, M, M> measNoise,
        cv::Matx<double, D, D> newComponentCov);
    LinearMeasurementModel<D, M> (cv::Matx<double, M, M> measNoise,
        cv::Matx<double, D, D> newComponentCov);
    LinearMeasurementModel<D, M> (const LinearMeasurementModel<D, M> & src);
    ~LinearMeasurementModel<D, M>(){};
    MeasurementModel<D, M> * copy() const;
    // Accessors
    cv::Matx<double, D, M> getMeasMatrixPseudoInverse() const {
      return mMeasMatrixPseudoInverse;
    };
    cv::Matx<double, D, D> getNewComponentCov() const {
      return mNewComponentCov;
    };
    // Methods
    cv::Vec<double, M> predict(cv::Vec<double, D> x);
    WeightedGaussian<D> invert(cv::Vec<double, M> z);
};

/*
class DisparitySpaceMeasurementModel: public MeasurementModel<4, 3> {
  protected:
    cv::Matx<double, 3, 3> mIntrinsicMatrix;
    cv::Vec<double, 3> mTranslation;
    double mRoll;
    double mPitch;
    double mYaw;
  public:
    DisparitySpaceMeasurementModel();
    DisparitySpaceMeasurementModel(cv::Matx<double, 3, 3> intrinsics,
        cv::Vec<double, 3> translation, double roll, double pitch,
        double yaw);
    cv::Vec<double, 3> predict(cv::Vec<double, 3> x);
    WeightedGaussian<4> invert(cv::Vec<double, 2> z);
    cv::Vec<double, 4> toDisparity(cv::Vec<double, 3> vecXYZ,
        cv::Matx<double, 3, 4> projMatrix);
    cv::Vec<double, 4> fromDisparity(cv::Vec<double, 3> vecXYD,
        cv::Matx<double, 3, 4> projMatrix);
    cv::Matx<double, 3, 4> getBaseProjectionMatrix();
    cv::Matx<double, 3, 4> getCurrentProjectionMatrix();
};
*/

/* Basic class to store GMPHD filter parameters */
class GMPHDFilterParams {
  public:
    double mProbSurvival;
    double mProbDetection;
    double mClutterDensity;
    double mMergeThreshold;
    double mTrimThreshold;
    int mTruncThreshold;
    GMPHDFilterParams(double ps = 0.99, double pd = 0.9, double k = 0.005,
        double mt = 1., double trit = 1e-3, int trut = 120){
      mProbSurvival = ps; mProbDetection = pd; mClutterDensity = k;
      mMergeThreshold = mt; mTrimThreshold = trit; mTruncThreshold = trut;}
};

template <int D, int M>
class GMPHDFilter {
  protected:
    MotionModel<D> * mMotionModel;
    MeasurementModel<D, M> * mMeasurementModel;
    GaussianMixture<D> mPHD;
    double mMultiObjectLikelihood;
    void predictLinear();
    void predictNonLinear();
    void updateLinear(std::vector<cv::Vec<double, M> > measurements);
    void updateNonLinear(std::vector<cv::Vec<double, M> > measurements);
  public:
    GMPHDFilter();
    GMPHDFilter(MotionModel<D> * motionModel,
        MeasurementModel<D, M> * measurementModel,
        double ps = 0.99, double pd = 0.9, double k = 0.005,
        double mt = 1., double trit = 1e-3, int trut = 120);
    GMPHDFilter(MotionModel<D> * motionModel,
        MeasurementModel<D, M> * measurementModel,
        GMPHDFilterParams params);
    GMPHDFilter (const GMPHDFilter<D, M> & src);
    ~GMPHDFilter();
    const GMPHDFilter & operator= (const GMPHDFilter & rhs);
    // Accessors
    GaussianMixture<D> getPHD() const {return mPHD;};
    MotionModel<D> * getMotionModel() const {return mMotionModel;};
    MeasurementModel<D, M> * getMeasurementModel() const {
      return mMeasurementModel;};
    double getMultiObjectLikelihood() const {return mMultiObjectLikelihood;};
    // Methods
    void predict();
    void update(std::vector<cv::Vec<double, M> >); // Includes birth
    double predictMeasurementLikelihood(
        std::vector<cv::Vec<double, M> > measurements);
    std::vector<cv::Vec<double, D> > getStateEstimate();
    // Parameters
    GMPHDFilterParams mParams;
};

// Attention: Using a bias is _not_ general enough except for very basic
// applications! However, for sensor drift estimation it is enough (for now)
template <int D, int M>
class GMPHDFilterParticle {
  public:
    // Members
    GMPHDFilter<D, M> mPHDFilter;
    double mWeight;
    cv::Vec<double, M> mBias;
    // Constructor
    GMPHDFilterParticle(){};
    GMPHDFilterParticle(GMPHDFilter<D, M> filter, double weight,
        cv::Vec<double, M> bias);
    GMPHDFilterParticle(const GMPHDFilterParticle & src);
    // Methods
    double predictMeasurementLikelihood(
        std::vector<cv::Vec<double, M> > measurements);
    void predict();
    void update(std::vector<cv::Vec<double, M> > measurements); 
    // Operators
    bool operator<(GMPHDFilterParticle<D, M> cmp) const {
      return mWeight < cmp.mWeight;};
    bool operator<=(GMPHDFilterParticle<D, M> cmp) const {
      return mWeight <= cmp.mWeight;};
    bool operator>(GMPHDFilterParticle<D, M> cmp) const {
      return mWeight > cmp.mWeight;};
    bool operator>=(GMPHDFilterParticle<D, M> cmp) const {
      return mWeight >= cmp.mWeight;};
};

// Building upon the previously constructed particle, this filter works only
// where only sensor bias wants to be corrected. It assumes a constant position
// motion model for the sensor. This should be made more general in the future
// to accept arbitrary motion/measurement models for the sensor state
template <int D, int M>
class CPPHDParticleFilter {
  public:
    CPPHDParticleFilter(unsigned int numComponents,
      cv::Matx<double, M, M> covariance, MotionModel<D> * motionModel,
      MeasurementModel<D, M> * measurementModel, GMPHDFilterParams params);
    std::vector<GMPHDFilterParticle<D, M> > mBelief;
    cv::Matx<double, M, M> mNoiseCovariance;
    void normalizeWeights();
    void basicResample();
    std::vector<cv::Vec<double, M> > regularizedBiases();
    void regularizedResample();
    void resample(std::vector<cv::Vec<double, M> > measurements);
    void predict();
    void update(std::vector<cv::Vec<double, M> > measurements);
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
  cv::Vec<double, 1> arg = (x - mean).t() * cov.inv() * (x - mean);
  double val = pow(2*M_PI, -d/2) * pow(cv::determinant(cov), -0.5) *
    exp( -0.5 * arg(0) );
  return val;
}

/* Draw a random sample from an empirical distribution characterized by
 * a vector of weights; The sum of weights must add up to one but this
 * is not verified */
int sampleEmpirical (std::vector<double> & weights) {
  double r = cv::randu<double>();
  double l = weights[0];
  int c = 0;
  while(l < r){
    l += weights[c];
    c++;
  }
  assert(c <= weights.size()); // No segfaults please
  return c;
}

/* This implementation is based on the method by Knuth. Perhaps inverse
 * transform sampling can be more efficient if an efficient form of the
 * Poisson CDF is used */
int samplePoisson(double lambda) {
  double L = exp(-lambda);
  double p = cv::randu<double>();
  int k = 1;
  while(p > L) {
    k++;
    p *= cv::randu<double>();
  }
  return --k;
}

/* This function randomly generates a number of vectors from a multivariate
 * normal distribution, with the given means and covariances. */

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

/* Returns the rotation matrix that transforms a coordinate frame by rotation
 * roll radians around the x axis, pitch radians around the y axis and finally
 * yaw radians around the z axis */
cv::Matx<double, 3, 3> RPYRotation(double roll, double pitch, double yaw) {
  cv::Matx<double, 3, 3> rotation;
  rotation <<
  cos(pitch) * cos(yaw),
  cos(roll) * sin(yaw) + sin(roll) * sin(pitch) * cos(yaw),
  sin(roll) * sin(yaw) - cos(roll) * sin(pitch) * cos(yaw),
  -cos(pitch)*sin(yaw),
  cos(roll) * cos(yaw) - sin(roll) * sin(pitch) * sin(yaw),
  sin(roll) * cos(yaw) + cos(roll) * sin(pitch) * sin(yaw),
  sin(pitch),
  -sin(roll) * cos(pitch),
  cos(roll) * cos(pitch) ;
  return rotation;
}

template <int M>
void meanAndCovariance(std::vector<cv::Vec<double, M> > samples,
    cv::Vec<double, M> & mean, cv::Matx<double, M, M> & cov) {
  typename std::vector<cv::Vec<double, M> >::iterator it;
  int N = samples.size();
  mean = mean * 0.;
  for (it = samples.begin(); it != samples.end(); ++it) {
    mean = mean + *it;
  }
  mean = (1./N) * mean;
  for (it = samples.begin(); it != samples.end(); ++it) {
    *it = *it - mean;
  }
  cov = cv::Matx<double, M, M>::zeros();
  for (it = samples.begin(); it != samples.end(); ++it) {
    cov = cov + ((*it) * (it->t()));
  }
  cov = (1./(N-1)) * cov;
  //cv::Mat demeanedSamples(M, N, CV_64F);
  //cv::Mat tempCov(M, M, CV_64F);
  //for (int i = 0; i < N; ++i) {
  //  for (int j = 0; j < M; ++j) {
  //    demeanedSamples.at<double>(j, i) = samples[i](j) - mean(j);
  //  }
  //}
  //tempCov = (1./(N-1)) * demeanedSamples * demeanedSamples.t();
  //for (int i = 0; i < M; ++i) {
  //  for (int j = 0; j < M; ++j) {
  //    cov(i, j) = tempCov.at<double>(i, j);
  //  }
  //}
  //cov = cv::Matx<double, M, M>(tempCov);
}

/* * * * * * * * * *
 * Implementation  *
 * * * * * * * * * */

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

template <int D>
GaussianMixture<D> :: GaussianMixture () {
}

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

// Provides a way to construct a motion model without arguments
template <int D>
MotionModel<D> :: MotionModel() {
  this->mIsLinear = true;
  this->mJacobian = cv::Matx<double, D, D>::zeros();
  this->mProcNoise = cv::Matx<double, D, D>::zeros();
}

template <int D>
MotionModel<D> :: MotionModel(const MotionModel<D> & src) {
  this->mIsLinear = src.isLinear();
  this->mJacobian = src.getJacobian();
  this->mProcNoise = src.getProcNoise();
}

template <int D, int M>
MeasurementModel<D, M> :: MeasurementModel() {
  this->mIsLinear = true;
  this->mJacobian = cv::Matx<double, M, D>::zeros();
  this->mMeasNoise = cv::Matx<double, M, M>::zeros();
}

template <int D, int M>
MeasurementModel<D, M> :: MeasurementModel (
    const MeasurementModel<D, M> & src) {
  this->mIsLinear = src.isLinear();
  this->mJacobian = src.getJacobian();
  this->mMeasNoise = src.getMeasNoise();
}

template <int D>
LinearMotionModel<D> :: LinearMotionModel (
    cv::Matx<double, D, D> dynamicsMatrix, cv::Matx<double, D, D> procNoise) {
  this->mIsLinear = true;
  this->mJacobian = dynamicsMatrix;
  this->mProcNoise = procNoise;
}

template <int D>
LinearMotionModel<D> :: LinearMotionModel (cv::Matx<double, D, D> procNoise) {
  this->mIsLinear = true;
  this->mJacobian = cv::Matx<double, D, D>::eye();
  this->mProcNoise = procNoise;
}

template <int D>
MotionModel<D> * LinearMotionModel<D> :: copy() const {
  MotionModel<D> * copiedModel = new LinearMotionModel<D>(*this);
  return copiedModel;
}

//template <int D>
//LinearMotionModel<D> :: LinearMotionModel (const LinearMotionModel<D> & src) {
//  this->mIsLinear = src.isLinear();
//  this->mJacobian = src.getJacobian();
//  this->mProcNoise = src.getProcNoise();
//}

template <int D>
cv::Vec<double, D> LinearMotionModel<D> :: predict (cv::Vec<double, D> x) {
  return (this->mJacobian) * x;
}

template <int D, int M>
LinearMeasurementModel<D, M> :: LinearMeasurementModel(
    cv::Matx<double, M, D> measurementMatrix, cv::Matx<double, M, M> measNoise,
    cv::Matx<double, D, D> newComponentCov) {
  this->mIsLinear = true;
  this->mJacobian = measurementMatrix;
  cv::Mat auxMat = (cv::Mat(this->mJacobian)).inv(cv::DECOMP_SVD);
  mMeasMatrixPseudoInverse = cv::Matx<double, D, M>(auxMat);
  this->mMeasNoise = measNoise;
  mNewComponentCov = newComponentCov;
}

template <int D, int M>
LinearMeasurementModel<D, M> :: LinearMeasurementModel(
    cv::Matx<double, M, M> measNoise, cv::Matx<double, D, D> newComponentCov) {
  this->mIsLinear = true;
  this->mJacobian = cv::Matx<double, M, D>::eye();
  mMeasMatrixPseudoInverse = (this->mJacobian).inv(cv::DECOMP_SVD);
  this->mMeasNoise = measNoise;
  mNewComponentCov = newComponentCov;
}

template <int D, int M>
LinearMeasurementModel<D, M> :: LinearMeasurementModel(
    const LinearMeasurementModel<D, M> & src) : MeasurementModel<D, M> (src) {
//  this->mIsLinear = src.isLinear();
//  this->mJacobian = src.getJacobian();
//  this->mMeasNoise = src.getMeasNoise();
  mMeasMatrixPseudoInverse = src.getMeasMatrixPseudoInverse();
  mNewComponentCov = src.getNewComponentCov();
}

template <int D, int M>
MeasurementModel<D, M> * LinearMeasurementModel<D, M> :: copy() const {
  MeasurementModel<D, M> * copiedModel = new LinearMeasurementModel(*this);
  return copiedModel;
}

template <int D, int M>
cv::Vec<double, M> LinearMeasurementModel<D, M> :: predict(cv::Vec<double, D> x) {
  return (this->mJacobian) * x;
}

template <int D, int M>
WeightedGaussian<D> LinearMeasurementModel<D, M> :: invert(cv::Vec<double, M> z) {
  cv::Vec<double, D> inverseMean = mMeasMatrixPseudoInverse * z;
  return WeightedGaussian<D>(1.0, inverseMean, mNewComponentCov);
}

/*
DisparitySpaceMeasurementModel :: DisparitySpaceMeasurementModel() {
  mIntrinsicMatrix = cv::Matx<double, 3, 3>::eye();
  mTranslation = cv::Vec<double, 3>::zeros();
  mRoll = 0; mPitch = 0; mYaw = 0;
}

DisparitySpaceMeasurementModel :: DisparitySpaceMeasurementModel(
    cv::Matx<double, 3, 3> intrinsics, cv::Vec<double, 3> translation,
    double roll, double pitch, double yaw) {
  mIntrinsicMatrix = intrinsics;
  mTranslation = translation;
  mRoll = roll; mPitch = pitch; mYaw = yaw;
}

cv::Matx<double, 3, 4>
DisparitySpaceMeasurementModel :: getCurrentProjectionMatrix() {
  cv::Matx<double, 3, 4> extrinsics = cv::Matx<double, 3, 4>::zeros();
  extrinsics(cv::Range(0, 3), cv::Range(0, 3)) = 
    RPYRotation(mRoll, mPitch, mYaw).t();
  extrinsics(cv::Range(0, 3), cv::Range(3, 4)) = 
    -1 * mTranslation;
  extrinsics(2, 3) = 1;
  cv::Matx<double, 3, 4> projMatrix = mIntrinsicMatrix * extrinsics;
  return projMatrix;
}
*/

template <int D, int M>
void GMPHDFilter<D, M> :: predictLinear() {
  typename std::vector<WeightedGaussian<D> >::iterator it;
  cv::Matx<double, D, D> J = mMotionModel->getJacobian();
  cv::Matx<double, D, D> Q = mMotionModel->getProcNoise();
  for (it = mPHD.mComponents.begin(); it != mPHD.mComponents.end(); ++it) {
    // Scale weight according to probability of survival
    it->setWeight( mParams.mProbSurvival * (it->getWeight()) );
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
  std::vector<cv::Matx<double, M, M> > S;
  std::vector<cv::Matx<double, D, M> > K;
  std::vector<cv::Matx<double, D, D> > P;
  cv::Matx<double, M, M> R = mMeasurementModel->getMeasNoise();
  cv::Matx<double, M, D> H = mMeasurementModel->getJacobian();
  cv::Matx<double, D, D> ID = cv::Matx<double, D, D>::eye();
  typename std::vector<WeightedGaussian<D> >::iterator it;
  typename std::vector<cv::Vec<double, M> >::iterator jt;
  // Compute exponential term of multi object likelihood
  double sumOfWeights = 0;
  for (it = prior.begin(); it != prior.end(); ++it) {
    sumOfWeights += it->getWeight();
  }
  mMultiObjectLikelihood = exp(-1 * mParams.mProbDetection * sumOfWeights);
  // Update previous weights and compute elements for update
  for (it = mPHD.mComponents.begin(); it != mPHD.mComponents.end(); ++it) {
    eta.push_back(H * it->getMean());
    S.push_back(R + H * (it->getCov()) * H.t());
    K.push_back((it->getCov()) * H.t() * S.back().inv());
    P.push_back((ID - K.back() * H)*(it->getCov()));
    // Scale weight according to probability of detection
    it->setWeight((1 - mParams.mProbDetection) * it->getWeight());
  }
  // Generate new Gaussian terms for each one of the measurements
  double nWeight;
  double denominator;
  // Birth weight is determined as a function of clutter density and a factor
  // of the trimming threshold.
  double birthWeight = 1.1 * mParams.mClutterDensity * mParams.mTrimThreshold
    / (1 - 1.1*mParams.mTrimThreshold);
  WeightedGaussian<D> birthComponent;
  cv::Vec<double, D> nMean;
  for (jt = measurements.begin(); jt != measurements.end(); ++jt) {
    denominator = birthWeight;
    // Add birth component
    birthComponent = mMeasurementModel->invert(*jt);
    birthComponent.setWeight(birthWeight);
    mPHD.mComponents.push_back(birthComponent);
    // Add components that account for every track's relation with received
    // measurement
    for (int i = 0; i < prior.size(); ++i) {
      nWeight = mParams.mProbDetection * (prior[i].getWeight()) *
          MVNormalPDF<M> (eta[i], S[i], *jt );
      denominator += nWeight;
      nMean = prior[i].getMean() + K[i]*(*jt - eta[i]);
      mPHD.mComponents.push_back(WeightedGaussian<D>(nWeight, nMean, P[i]));
    }
    // Complete the denominator for the new Gaussian terms and apply it
    denominator += mParams.mClutterDensity;
    mMultiObjectLikelihood *= denominator;
    for (int i = mPHD.mComponents.size() - prior.size() - 1;
        i < mPHD.mComponents.size(); ++i) {
      mPHD.mComponents[i].setWeight(
          mPHD.mComponents[i].getWeight() / denominator);
    }
  }
}

template <int D, int M>
void GMPHDFilter<D, M> :: updateNonLinear(
    std::vector<cv::Vec<double, M> > measurements) {
  // UK PHD Update
  // TODO Implement
}

template <int D, int M>
GMPHDFilter<D, M> :: GMPHDFilter() {
  mMotionModel = NULL;
  mMeasurementModel = NULL;
  mMultiObjectLikelihood = 0;
}

template <int D, int M>
GMPHDFilter<D, M> :: GMPHDFilter(MotionModel<D> * motionModel,
    MeasurementModel<D, M> * measurementModel, double ps, double pd,
    double k, double mt, double trit, int trut) {
  mMotionModel = motionModel;
  mMeasurementModel = measurementModel;
  mParams = GMPHDFilterParams(ps, pd, k, mt, trit, trut);
  mMultiObjectLikelihood = 0;
}

template <int D, int M>
GMPHDFilter<D, M> :: GMPHDFilter(MotionModel<D> * motionModel,
    MeasurementModel<D, M> * measurementModel,
    GMPHDFilterParams params) {
  mMotionModel = motionModel;
  mMeasurementModel = measurementModel;
  mParams = params;
  mMultiObjectLikelihood = 0;
}

template <int D, int M>
GMPHDFilter<D, M> :: GMPHDFilter(const GMPHDFilter<D, M> & src) {
  //mMotionModel = (src.getMotionModel())->copy();
  MotionModel<D> * intMotionModel = src.getMotionModel();
  mMotionModel = intMotionModel->copy();
  //mMeasurementModel = (src.getMeasurementModel())->copy();
  MeasurementModel<D, M> * intMeasModel = src.getMeasurementModel();
  mMeasurementModel = intMeasModel->copy();
  mPHD = src.getPHD();
  mMultiObjectLikelihood = src.getMultiObjectLikelihood();
  mParams = src.mParams;
}

template <int D, int M>
GMPHDFilter<D, M> :: ~GMPHDFilter() {
  delete mMotionModel;
  delete mMeasurementModel;
}

template <int D, int M>
const GMPHDFilter<D, M> & GMPHDFilter<D, M> :: operator= (
    const GMPHDFilter & rhs) {
  if (this != &rhs) {
    mMotionModel = rhs.getMotionModel()->copy();
    mMeasurementModel = rhs.getMeasurementModel()->copy();
    mPHD = rhs.getPHD();
    mMultiObjectLikelihood = rhs.getMultiObjectLikelihood();
    mParams = rhs.mParams;
  }
  return *this;
}


template <int D, int M>
void GMPHDFilter<D, M> :: predict() {
  if (mMotionModel != NULL) {
    if (mMotionModel->isLinear()) {
      predictLinear();
    } else {
      predictNonLinear();
    }
  } else {
    std::cout << "Uninitialized motion model\n";
    exit(1);
  }
}

template <int D, int M>
void GMPHDFilter<D, M> :: update(
    std::vector<cv::Vec<double, M> > measurements) {
  if (mMeasurementModel != NULL) {
    if (mMeasurementModel->isLinear()) {
      updateLinear(measurements);
    } else {
      updateNonLinear(measurements);
    }
    mPHD.merge(mParams.mMergeThreshold);
    mPHD.trim(mParams.mTrimThreshold);
    mPHD.truncate(mParams.mTruncThreshold);
  } else {
    std::cout << "Uninitialized measurement model\n";
    exit(1);
  }
}

template <int D, int M>
double GMPHDFilter<D, M> :: predictMeasurementLikelihood(
    std::vector<cv::Vec<double, M> > measurements) {
  double likelihood;
  // Copy of prior weights
  std::vector<double> priorWeights;
  // Components for updated terms
  std::vector<cv::Vec<double, M> > eta;
  std::vector<cv::Matx<double, M, M> > S;
  cv::Matx<double, M, M> R = mMeasurementModel->getMeasNoise();
  cv::Matx<double, M, D> H = mMeasurementModel->getJacobian();
  cv::Matx<double, D, D> ID = cv::Matx<double, D, D>::eye();
  typename std::vector<WeightedGaussian<D> >::iterator it;
  typename std::vector<cv::Vec<double, M> >::iterator jt;
  // Compute exponential term of multi object likelihood
  double sumOfWeights = 0;
  for (it = mPHD.mComponents.begin(); it != mPHD.mComponents.end(); ++it) {
    sumOfWeights += it->getWeight();
    priorWeights.push_back(it->getWeight());
  }
  likelihood = exp(-1 * mParams.mProbDetection * sumOfWeights);
  // Update previous weights and compute elements for update
  for (it = mPHD.mComponents.begin(); it != mPHD.mComponents.end(); ++it) {
    eta.push_back(H * it->getMean());
    S.push_back(R + H * (it->getCov()) * H.t());
  }
  // Generate new Gaussian terms for each one of the measurements
  double nWeight;
  double denominator;
  // Birth weight is determined as a function of clutter density and a factor
  // of the trimming threshold.
  double birthWeight = 1.1 * mParams.mClutterDensity * mParams.mTrimThreshold
    / (1 - 1.1*mParams.mTrimThreshold);
  WeightedGaussian<D> birthComponent;
  cv::Vec<double, D> nMean;
  for (jt = measurements.begin(); jt != measurements.end(); ++jt) {
    // Add the birth weight to the likelihood factor
    denominator = birthWeight;
    for (int i = 0; i < priorWeights.size(); ++i) {
      // Add the updated weight to the likelihood factor
      denominator += mParams.mProbDetection * (priorWeights[i]) *
          MVNormalPDF<M> (eta[i], S[i], *jt );
    }
    // Add the clutter density to the factor and integrate with the likelihood
    // the likelihood
    denominator += mParams.mClutterDensity;
    likelihood *= denominator;
  }
  return likelihood;
}

template <int D, int M>
std::vector<cv::Vec<double, D> > GMPHDFilter<D, M> :: getStateEstimate () {
  typename std::vector<WeightedGaussian<D> >::iterator it;
  std::vector<cv::Vec<double, D> > stateEstimate;
  for (it = mPHD.mComponents.begin(); it != mPHD.mComponents.end(); ++it) {
    if (it->getWeight() > 0.5) {
      stateEstimate.push_back(it->getMean());
    }
  }
  return stateEstimate;
}

template <int D, int M>
GMPHDFilterParticle<D, M> :: GMPHDFilterParticle(GMPHDFilter<D, M> filter,
    double weight, cv::Vec<double, M> bias) {
  mPHDFilter = filter;
  mWeight = weight;
  mBias = bias;
}

template <int D, int M>
GMPHDFilterParticle <D, M> :: GMPHDFilterParticle(
    const GMPHDFilterParticle &src) {
  mPHDFilter = src.mPHDFilter;
  mWeight = src.mWeight;
  mBias = src.mBias;
}

template <int D, int M>
double GMPHDFilterParticle<D, M> :: predictMeasurementLikelihood(
    std::vector<cv::Vec<double, M> > measurements) {
  std::vector<cv::Vec<double, M> > biasedMeasurements;
  typename std::vector<cv::Vec<double, M> >::iterator it;
  for (it = measurements.begin(); it != measurements.end(); ++it) {
    biasedMeasurements.push_back(*it - mBias);
  }
  return mPHDFilter.predictMeasurementLikelihood(biasedMeasurements);
}

template <int D, int M>
void GMPHDFilterParticle<D, M> :: predict() {
  mPHDFilter.predict();
}

template <int D, int M>
void GMPHDFilterParticle<D, M> :: update(
    std::vector<cv::Vec<double, M> > measurements) {
  std::vector<cv::Vec<double, M> > biasedMeasurements;
  typename std::vector<cv::Vec<double, M> >::iterator it;
  for (it = measurements.begin(); it != measurements.end(); ++it) {
    biasedMeasurements.push_back(*it - mBias);
  }
  mPHDFilter.update(biasedMeasurements);
}

template <int D, int M>
CPPHDParticleFilter<D, M> :: CPPHDParticleFilter(unsigned int numComponents,
    cv::Matx<double, M, M> noiseCovariance, MotionModel<D> * motionModel,
    MeasurementModel<D, M> * measurementModel, GMPHDFilterParams params){
  mNoiseCovariance = noiseCovariance;
  cv::Vec<double, M> zeros;
  for (int i = 0; i < M; ++i) {
    zeros(i) = 0;
  }
  double initialWeight = 1./numComponents;
  MotionModel<D> * motionModelCopy = NULL;
  MeasurementModel<D, M> * measurementModelCopy = NULL;
  GMPHDFilter<D, M> filterCopy;
  GMPHDFilterParticle<D, M> particleCopy;
  for (int i = 0; i < numComponents; ++i) {
    motionModelCopy = motionModel->copy();
    measurementModelCopy = measurementModel->copy();
    filterCopy =  GMPHDFilter<D, M> (
        motionModelCopy, measurementModelCopy, params);
    particleCopy = GMPHDFilterParticle<D, M>(filterCopy,
        initialWeight, zeros);
    mBelief.push_back(particleCopy);
  }
}

template <int D, int M>
void CPPHDParticleFilter<D, M> :: normalizeWeights() {
  typename std::vector<GMPHDFilterParticle<D, M> >::iterator it;
  double sumOfWeights = 0;
  for (it = mBelief.begin(); it != mBelief.end(); ++it) {
    sumOfWeights += it->mWeight;
  }
  for (it = mBelief.begin(); it != mBelief.end(); ++it) {
    it->mWeight /= sumOfWeights;
  }
}

// Roulette resampling for the Particle PHD Filter
template <int D, int M>
void CPPHDParticleFilter<D, M> :: basicResample() {
  typename std::vector<GMPHDFilterParticle<D, M> >::iterator it;
  double sumOfWeights = 0;
  std::vector<GMPHDFilterParticle<D, M> > resampledBelief;
  std::vector<GMPHDFilterParticle<D, M> > newBelief;
  double location = cv::randu<double>();
  double cumsum = 0;
  double increment = 1./mBelief.size();
  int j = 0;
  for (int i = 0; i < mBelief.size(); ++i) {
    while (cumsum < location) {
      cumsum += mBelief[j].mWeight;
      ++j;
      if (j >= mBelief.size()){
        j = 0;
        cumsum = 0;
        location -= 1;
      }
    }
    newBelief.push_back(mBelief[j]);
    location += increment;
  }
  mBelief = newBelief;
}

template <int D, int M>
void CPPHDParticleFilter<D, M> :: predict() {
  cv::Vec<double, M> zeros;
  for (int i = 0; i < M; ++i) {
    zeros(i) = 0;
  }
  std::vector<cv::Vec<double, M> > noise = sampleMVGaussian(zeros,
      mNoiseCovariance, mBelief.size());
  for (int i = 0; i < mBelief.size(); ++i) {
    mBelief[i].mBias += noise[i];
    mBelief[i].predict();
  }
}

template <int D, int M>
std::vector<cv::Vec<double, M> > CPPHDParticleFilter<D, M>::regularizedBiases() {
  std::vector<cv::Vec<double, M> > biases;
  std::vector<cv::Vec<double, M> > whitenedBiases;
  std::vector<cv::Vec<double, M> > sampledBiases;
  std::vector<double> normalizedWeights;
  typename std::vector<cv::Vec<double, M> >::iterator it;
  typename std::vector<GMPHDFilterParticle<D, M> >::iterator jt;
  cv::Vec<double, M> sampleMean;
  cv::Matx<double, M, M> sampleCov;
  cv::Vec<double, M> zeros;
  for (int i = 0; i < M; ++i) { zeros(i) = 0; }
  double sumOfWeights;
  for (jt = mBelief.begin(); jt != mBelief.end(); ++jt) {
    biases.push_back(jt->mBias);
    normalizedWeights.push_back(jt->mWeight);
    sumOfWeights += jt->mWeight;
  }
  for (std::vector<double>::iterator kt = normalizedWeights.begin();
      kt != normalizedWeights.end(); ++kt) {
    *kt = *kt/sumOfWeights;
  }
  meanAndCovariance(biases, sampleMean, sampleCov);
  cv::Matx<double, M, M> decomposedCovariance;
  cv::Matx<double, M, M> invDecomposedCovariance;
  bool decomposed = cholesky<M>(sampleCov, decomposedCovariance);
  invDecomposedCovariance = decomposedCovariance.inv();
  for (it = biases.begin(); it != biases.end(); ++it) {
    whitenedBiases.push_back(invDecomposedCovariance * (*it));
  }
  // Taken from SMC Methods in Practice (Doucet 2001)
  double bandWidth = pow(4/(M+2), 1/(M+4))*pow(mBelief.size(),-1/(M+4));
  std::vector<cv::Vec<double, M> > noise = sampleMVGaussian<M>(
      zeros, cv::Matx<double, M, M>::eye(), mBelief.size());
  int index;
  // Copy copy copycat!
  double location = cv::randu<double>();
  double cumsum = 0;
  double increment = 1./mBelief.size();
  int j = 0;
  for (int i = 0; i < mBelief.size(); ++i) {
    while (cumsum < location) {
      cumsum += normalizedWeights[j];
      ++j;
      if (j >= mBelief.size()){
        j = 0;
        cumsum = 0;
        location -= 1;
      }
    }
    sampledBiases.push_back(
        mBelief[j].mBias + bandWidth*decomposedCovariance*noise[i]);
    location += increment;
  }
  // Copy copy copyquat
  //for (int i = 0; i < mBelief.size(); ++i) {
  //  index = sampleEmpirical(normalizedWeights);
  //  sampledBiases.push_back(
  //      mBelief[i].mBias + bandWidth*decomposedCovariance*noise[i]);
    //sampledBiases.push_back( (1/bandWidth) *
    //    decomposedCovariance * ( whitenedBiases[index] + noise[i] ) );
  //}
  return sampledBiases;
}

template <int D, int M>
void CPPHDParticleFilter<D, M> :: regularizedResample() {
  std::vector<cv::Vec<double, M> > regularizedBias = regularizedBiases();
  typename std::vector<GMPHDFilterParticle<D, M> >::iterator it;
  typename std::vector<cv::Vec<double, M> >::iterator jt;
  for (it = mBelief.begin(), jt = regularizedBias.begin();
       it != mBelief.end(); ++it, ++jt) {
    it->mBias = *jt;
  }
}

// This implements a simplified version (no adaptive tempering) version of
// a particle filter with importance sampling with sequential correction 
template <int D, int M>
void CPPHDParticleFilter<D, M> :: resample(
    std::vector<cv::Vec<double, M> > measurements) {
  // This struct can be used to sort a vector of indices by comparing
  // elements within the provided vector of doubles
  typename std::vector<GMPHDFilterParticle<D, M> >::iterator it;
  typename std::vector<cv::Vec<double, M> >::iterator jt;
  std::vector<double>::iterator kt;
  std::vector<cv::Vec<double, M> > weightedSample;
  std::vector<cv::Vec<double, M> > sampledBiases;
  std::vector<double> oldLikelihoods;
  cv::Vec<double, M> sampleMean;
  cv::Matx<double, M, M> sampleCov;
  cv::Matx<double, M, M> scaledCov;
  cv::Vec<double, M> small;
  cv::Vec<double, M> large;
  double covarianceProportion =2.;
  int numberOfIterations = 15;
  double likelihoodPower = 0;
  double acceptanceProbability;
  cv::Matx<double, M, M> unusedMatrix;
  double acceptProb;
  cv::Vec<double, M> oldBias;
  double newLikelihood;
  double proposalOld, proposalNew;
  for (int i = 0; i < numberOfIterations; ++i) {
    oldLikelihoods.clear();
    likelihoodPower += 1./double(numberOfIterations);
    for (it = mBelief.begin(); it != mBelief.end(); ++it) {
      oldLikelihoods.push_back(it->predictMeasurementLikelihood(measurements));
      it->mWeight = pow(oldLikelihoods.back(), likelihoodPower);
    }
    normalizeWeights();
    sampledBiases = regularizedBiases();
    for (it = mBelief.begin(), jt = sampledBiases.begin(),
        kt = oldLikelihoods.begin(); it != mBelief.end(); ++it, ++jt, ++kt) {
      oldBias = it->mBias;
      it->mBias = *jt;
      newLikelihood = it->predictMeasurementLikelihood(measurements);
      acceptProb = std::min(1., 0.25*newLikelihood/(*kt));
      if ((*kt != 0) && cv::randu<double>() > acceptProb){
        it->mBias = oldBias;
      }
    }
    for (int i = 0; i < mBelief.size(); ++i) {
      small[0] = std::min(small[0], mBelief[i].mBias[0]);
      small[1] = std::min(small[1], mBelief[i].mBias[1]);
      large[0] = std::max(large[0], mBelief[i].mBias[0]);
      large[1] = std::max(large[1], mBelief[i].mBias[1]);
    }
  }
}
template <int D, int M>
void CPPHDParticleFilter<D, M> :: update(
    std::vector<cv::Vec<double, M> > measurements) {
  typename std::vector<GMPHDFilterParticle<D, M> >::iterator it;
  for (it = mBelief.begin(); it != mBelief.end(); ++it) {
    it->update(measurements);
  }
}
