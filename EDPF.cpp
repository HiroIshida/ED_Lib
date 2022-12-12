#include "EDPF.h"

using namespace cv;
using namespace std;

EDPF::EDPF(const int _width, const int _height)
: ED(_width, _height)
{
  prealloc();
}
/*
EDPF::EDPF(Mat srcImage) : ED(srcImage, PREWITT_OPERATOR, 11, 3)
{
  // Validate Edge Segments
  const auto start_tick = getTickCount();
  sigma /= 2.5;
  GaussianBlur(srcImage, smoothImage, Size(), sigma);  // calculate kernel from sigma
  const auto gaussian_blur_tick = getTickCount();
  lastEDPFProfile.gaussian_blur = (gaussian_blur_tick - start_tick) / getTickFrequency();

  validateEdgeSegments();
  const auto validate_edge_segments_tick = getTickCount();
  lastEDPFProfile.validate_edge_segments = (validate_edge_segments_tick - gaussian_blur_tick) / getTickFrequency();
}
*/
EDPF::EDPF(Mat srcImage) : ED(srcImage.cols, srcImage.rows)
{
  prealloc();
  process(srcImage);
}
EDPF::EDPF(ED obj) : ED(obj)
{
  // Validate Edge Segments
  sigma /= 2.5;
  GaussianBlur(srcImage, smoothImage, Size(), sigma);  // calculate kernel from sigma

  validateEdgeSegments();
}

EDPF::EDPF(EDColor obj) : ED(obj) {}

void EDPF::prealloc()
{
  H.reserve(MAX_GRAD_VALUE);
}

void EDPF::process(cv::Mat _srcImage)
{
  ED::process(_srcImage, PREWITT_OPERATOR, 11, 3);
  
  // Validate Edge Segments
  const auto start_tick = getTickCount();
  sigma /= 2.5;
  GaussianBlur(srcImage, smoothImage, Size(), sigma);  // calculate kernel from sigma
  const auto gaussian_blur_tick = getTickCount();
  lastEDPFProfile.gaussian_blur = (gaussian_blur_tick - start_tick) / getTickFrequency();

  validateEdgeSegments();
  const auto validate_edge_segments_tick = getTickCount();
  lastEDPFProfile.validate_edge_segments = (validate_edge_segments_tick - gaussian_blur_tick) / getTickFrequency();
}

void EDPF::validateEdgeSegments()
{
  divForTestSegment = 2.25;            // Some magic number :-)
  memset(edgeImg, 0, width * height);  // clear edge image

  ComputePrewitt3x3();

  // Compute np: # of segment pieces
#if 1
  // Does this underestimate the number of pieces of edge segments?
  // What's the correct value?
  np = 0;
  for (int i = 0; i < getSegmentNo(); i++)
  {
    int len = segmentPoints[i].size();
    np += (len * (len - 1)) / 2;
  }  // end-for

  //  np *= 32;
#elif 0
  // This definitely overestimates the number of pieces of edge segments
  int np = 0;
  for (int i = 0; i < getSegmentNo(); i++)
  {
    np += segmentPoints[i].size();
  }  // end-for
  np = (np * (np - 1)) / 2;
#endif

  // Validate segments
  for (int i = 0; i < getSegmentNo(); i++)
  {
    TestSegment(i, 0, segmentPoints[i].size() - 1);
  }  // end-for

  ExtractNewSegments();
}

void EDPF::ComputePrewitt3x3()
{
  const cv::Mat kernel = (cv::Mat_<int>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
  cv::Mat gxImageSigned, gyImageSigned;
  cv::filter2D(srcImage, gxImageSigned, CV_16SC1, kernel.t());
  cv::filter2D(srcImage, gyImageSigned, CV_16SC1, kernel);
  gradImage = cv::abs(gxImageSigned) + cv::abs(gyImageSigned);
  gradImage.col(0).setTo(0);
  gradImage.col(gradImage.cols - 1).setTo(0);
  gradImage.row(0).setTo(0);
  gradImage.row(gradImage.rows - 1).setTo(0);
  gradImg = (short *)gradImage.data;

  double max_grad_value = static_cast<double>(MAX_GRAD_VALUE);
  cv::minMaxLoc(gradImage, nullptr, &max_grad_value);
  std::vector<int> grads(static_cast<int>(max_grad_value) + 1, 0);
  for (int i = 0; i < gradImage.total(); ++i)
  {
    grads[gradImg[i]]++;
  }

  // Compute probability function H
  const int size = (width - 2) * (height - 2);
  
  for (int i = grads.size() - 1; i > 0; i--) grads[i - 1] += grads[i];

  H.clear();
  H.resize(grads.size(), 0);
  for (int i = 0; i < grads.size(); i++) H[i] = (double)grads[i] / ((double)size);
}

//----------------------------------------------------------------------------------
// Resursive validation using half of the pixels as suggested by DMM algorithm
// We take pixels at Nyquist distance, i.e., 2 (as suggested by DMM)
//
void EDPF::TestSegment(int i, int index1, int index2)
{
  int chainLen = index2 - index1 + 1;
  if (chainLen < minPathLen) return;

  // Test from index1 to index2. If OK, then we are done. Otherwise, split into two and
  // recursively test the left & right halves

  // First find the min. gradient along the segment
  int minGrad = 1 << 30;
  int minGradIndex;
  for (int k = index1; k <= index2; k++)
  {
    int r = segmentPoints[i][k].y;
    int c = segmentPoints[i][k].x;
    if (gradImg[r * width + c] < minGrad)
    {
      minGrad = gradImg[r * width + c];
      minGradIndex = k;
    }
  }  // end-for

  // Compute nfa
  double nfa = NFA(H[minGrad], (int)(chainLen / divForTestSegment));

  if (nfa <= EPSILON)
  {
    for (int k = index1; k <= index2; k++)
    {
      int r = segmentPoints[i][k].y;
      int c = segmentPoints[i][k].x;

      edgeImg[r * width + c] = 255;
    }  // end-for

    return;
  }  // end-if

  // Split into two halves. We divide at the point where the gradient is the minimum
  int end = minGradIndex - 1;
  while (end > index1)
  {
    int r = segmentPoints[i][end].y;
    int c = segmentPoints[i][end].x;

    if (gradImg[r * width + c] <= minGrad)
      end--;
    else
      break;
  }  // end-while

  int start = minGradIndex + 1;
  while (start < index2)
  {
    int r = segmentPoints[i][start].y;
    int c = segmentPoints[i][start].x;

    if (gradImg[r * width + c] <= minGrad)
      start++;
    else
      break;
  }  // end-while

  TestSegment(i, index1, end);
  TestSegment(i, start, index2);
}

//----------------------------------------------------------------------------------------------
// After the validation of the edge segments, extracts the valid ones
// In other words, updates the valid segments' pixel arrays and their lengths
//
void EDPF::ExtractNewSegments()
{
  // vector<Point> *segments = &segmentPoints[getSegmentNo()];
  vector<vector<Point> > validSegments;
  int noSegments = 0;

  for (int i = 0; i < getSegmentNo(); i++)
  {
    int start = 0;
    while (start < segmentPoints[i].size())
    {
      while (start < segmentPoints[i].size())
      {
        int r = segmentPoints[i][start].y;
        int c = segmentPoints[i][start].x;

        if (edgeImg[r * width + c]) break;
        start++;
      }  // end-while

      int end = start + 1;
      while (end < segmentPoints[i].size())
      {
        int r = segmentPoints[i][end].y;
        int c = segmentPoints[i][end].x;

        if (edgeImg[r * width + c] == 0) break;
        end++;
      }  // end-while

      int len = end - start;
      if (len >= 10)
      {
        // A new segment. Accepted only only long enough (whatever that means)
        // segments[noSegments].pixels = &map->segments[i].pixels[start];
        // segments[noSegments].noPixels = len;
        validSegments.push_back(vector<Point>());
        vector<Point> subVec(&segmentPoints[i][start], &segmentPoints[i][end - 1]);
        validSegments[noSegments] = subVec;
        noSegments++;
      }  // end-else

      start = end + 1;
    }  // end-while
  }    // end-for

  // Copy to ed
  segmentPoints = validSegments;
}

//---------------------------------------------------------------------------
// Number of false alarms code as suggested by Desolneux, Moisan and Morel (DMM)
//
double EDPF::NFA(double prob, int len)
{
  double nfa = np;
  for (int i = 0; i < len && nfa > EPSILON; i++) nfa *= prob;

  return nfa;
}

EDPF::Profile EDPF::getLastEDPFProfile() const { return lastEDPFProfile; }
