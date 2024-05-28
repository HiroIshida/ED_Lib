/**************************************************************************************************************
 * Edge Drawing (ED) and Edge Drawing Parameter Free (EDPF) source codes.
 * Copyright (C) Cihan Topal & Cuneyt Akinlar
 * E-mails of the authors:  cihantopal@gmail.com, cuneytakinlar@gmail.com
 *
 * Please cite the following papers if you use Edge Drawing library:
 *
 * [1] C. Topal and C. Akinlar, “Edge Drawing: A Combined Real-Time Edge and Segment Detector,”
 *     Journal of Visual Communication and Image Representation, 23(6), 862-872,
 *DOI: 10.1016/j.jvcir.2012.05.004 (2012).
 *
 * [2] C. Akinlar and C. Topal, “EDPF: A Real-time Parameter-free Edge Segment Detector with a False
 *Detection Control,” International Journal of Pattern Recognition and Artificial Intelligence,
 *26(1), DOI: 10.1142/S0218001412550026 (2012).
 **************************************************************************************************************/

#ifndef _ED_
#define _ED_

#include <deque>

#include <opencv2/opencv.hpp>
#include "EDColor.h"

/// Special defines
#define EDGE_VERTICAL 1
#define EDGE_HORIZONTAL 2

#define ANCHOR_PIXEL 254
#define EDGE_PIXEL 255

#define LEFT 1
#define RIGHT 2
#define UP 3
#define DOWN 4

enum GradientOperator
{
  PREWITT_OPERATOR = 101,
  SOBEL_OPERATOR = 102,
  SCHARR_OPERATOR = 103,
  LSD_OPERATOR = 104
};

struct StackNode
{
  int r, c;    // starting pixel
  int parent;  // parent chain (-1 if no parent)
  int dir;     // direction where you are supposed to go
};

// Used during Edge Linking
struct Chain
{
  int dir;            // Direction of the chain
  int len;            // # of pixels in the chain
  int parent;         // Parent of this node (-1 if no parent)
  int children[2];    // Children of this node (-1 if no children)
  cv::Point *pixels;  // Pointer to the beginning of the pixels array
};

class ED
{
 public:
  struct Profile
  {
    double initialize;
    double gaussian_blur;
    double compute_gradient;
    double compute_anchor_points;
    double join_anchor_points_using_sorted_anchors;
    // in JoinAnchorPointsUsingSortedAnchors method
    double join_anchor_points_alloc;
    double sort_anchors_by_grad_value;
  };

  ED(const int _width, const int _height);
  ED(cv::Mat _srcImage, GradientOperator _op = PREWITT_OPERATOR, int _gradThresh = 20,
     int _anchorThresh = 0, int _scanInterval = 1, int _minPathLen = 10, int _kSize = 5,
     double _sigma = 1.0, bool _sumFlag = true);
  ED(const ED &cpyObj);
  ED(short *gradImg, uchar *dirImg, int _width, int _height, int _gradThresh, int _anchorThresh,
     int _scanInterval = 1, int _minPathLen = 10, bool selectStableAnchors = true);
  ED(EDColor &cpyObj);
  ED();

  void prealloc(const int _width, const int _height);

  void process(cv::Mat _srcImage, GradientOperator _op = PREWITT_OPERATOR, int _gradThresh = 20,
               int _anchorThresh = 0, int _scanInterval = 1, int _minPathLen = 10,
               int _kSize = 5, double _sigma = 1.0, bool _sumFlag = true);
  int getWidth() const { return width; }
  int getHeight() const { return height; }

  cv::Mat getEdgeImage();
  cv::Mat getAnchorImage();
  cv::Mat getSmoothImage();
  cv::Mat getGradImage();
  cv::Mat getDirImage();

  int getSegmentNo();
  int getAnchorNo();

  std::vector<cv::Point> getAnchorPoints();
  std::vector<std::vector<cv::Point>> getSegments();
  std::vector<std::vector<cv::Point>> getSortedSegments();

  cv::Mat drawParticularSegments(std::vector<int> list);

  Profile getLastEDProfile() const;

 protected:
  int width;   // width of source image
  int height;  // height of source image
  uchar *srcImg;
  std::vector<std::vector<cv::Point>> segmentPoints;
  double sigma;  // Gaussian sigma
  cv::Mat smoothImage;
  uchar *edgeImg;    // pointer to edge image data
  uchar *smoothImg;  // pointer to smoothed image data
  int minPathLen;
  cv::Mat srcImage;
  Profile lastEDProfile;

  void ComputeGradient();
  void ComputeAnchorPoints();
  void JoinAnchorPointsUsingSortedAnchors();
  void sortAnchorsByGradValue();
  void sortAnchorsByGradValue1(std::vector<int> &A);

  static int LongestChain(std::vector<Chain> &chains, int root);
  static int RetrieveChainNos(std::vector<Chain> &chains, int root, std::vector<int> &chainNos);

  std::vector<cv::Point> takePointVectorFromPool();
  void returnPointVectorToPool(std::vector<cv::Point> point_vec);

  std::vector<cv::Point> anchorPoints;

  cv::Mat edgeImage;
  cv::Mat dirImage;
  cv::Mat gradImage;

  // Buffer with the same size as srcImage
  // Use these locally and
  // don't share data among functions with these
  cv::Mat buffer0;
  cv::Mat buffer1;
  cv::Mat buffer2;
  cv::Mat buffer3;

  uchar *dirImg;   // pointer to direction image data
  short *gradImg;  // pointer to gradient image data

  GradientOperator op;  // operation used in gradient calculation
  int gradThresh;       // gradient threshold
  int anchorThresh;     // anchor point threshold
  int scanInterval;
  bool sumFlag;

  // cached data used in JoinAnchorPointsUsingSortedAnchors
  std::vector<int> chainNos;
  std::vector<cv::Point> pixels;
  std::vector<StackNode> stack;
  std::vector<Chain> chains;

  // pool for cv::Point vector
  std::deque<std::vector<cv::Point>> m_point_vector_pool;
};

#endif
