#pragma once

#pragma warning(push, 0)
#pragma warning(pop)
#pragma warning(disable: 4251)

#ifndef DllCoreExport
#ifdef DK_CORE_DLL_EXPORT
#define DllCoreExport Q_DECL_EXPORT
#elif DK_DLL_IMPORT
#define DllCoreExport Q_DECL_IMPORT
#else
#define DllCoreExport Q_DECL_IMPORT
#endif
#endif

#include <QImage>
#include "DKImgHoleResults.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

namespace nmc {

class DllCoreExport DKImgPretreatment {

public:
	enum PRE_MODE {
		AUTO_MODE = 0,
		TIFF_MODE,
	};
	DKImgPretreatment(QImage orig,int mode = PRE_MODE::AUTO_MODE);
	DKImgHoleResults* getResult();
private:
	void covGrayImg();
	void getMinMaxGrayValue(int& minGrayValue, int& maxGrayValue);
	void binarization(int T);
	int   getIterationHresholdValue(int minGrayValue, int maxGrayValue);
	void goToOpenCv();
	void filterContours(std::vector<std::vector<cv::Point>> contours);
	int getXYColor(int x, int y, int r0);
	int getBinarizationT();
	void adaptiveThereshold();
private:
	DKImgHoleResults* mResult;
	int mMode;
};

}
