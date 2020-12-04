//
// Created by bruce on 2020/9/1.
//

#include "track.h"
#include "tracker.h"
#include "matrix.h"
#include "munkres.h"

float CalculateIou(const xsk::Rect &det, const Tracker &track)
{
    auto trk = track.GetStateAsBbox();
    // get min/max points
    auto xx1 = std::max(det.tl().x, trk.tl().x);
    auto yy1 = std::max(det.tl().y, trk.tl().y);
    auto xx2 = std::min(det.br().x, trk.br().x);
    auto yy2 = std::min(det.br().y, trk.br().y);
    auto w = std::max(0, xx2 - xx1);
    auto h = std::max(0, yy2 - yy1);

    // calculate area of intersection and union
    auto det_area = float(det.area());
    auto trk_area = float(trk.area());
    auto intersection_area = w * h;
    float union_area = det_area + trk_area - intersection_area;
    auto iou = intersection_area / union_area;
    return iou;
}

void HungarianMatching(const std::vector<std::vector<float>> &iou_matrix,
                       size_t nrows, size_t ncols,
                       std::vector<std::vector<float>> &association)
{
    Matrix<float> matrix(nrows, ncols);
    // Initialize matrix with IOU values
    for (size_t i = 0; i < nrows; i++)
    {
        for (size_t j = 0; j < ncols; j++)
        {
            // Multiply by -1 to find max cost
            if (iou_matrix[i][j] != 0)
            {
                matrix(i, j) = -iou_matrix[i][j];
            }
            else
            {
                // TODO: figure out why we have to assign value to get correct result
                matrix(i, j) = 1.0f;
            }
        }
    }

    // Apply Kuhn-Munkres algorithm to matrix.
    Munkres<float> m;
    m.solve(matrix);

    for (size_t i = 0; i < nrows; i++)
    {
        for (size_t j = 0; j < ncols; j++)
        {
            association[i][j] = matrix(i, j);
        }
    }
}

void AssociateDetectionsToTrackers(const std::vector<xsk::Rect> &detection,
                                   std::map<int, Tracker> &tracks,
                                   std::map<int, xsk::Rect> &matched,
                                   std::vector<xsk::Rect> &unmatched_det,
                                   float iou_threshold = 0.3)
{

    // Set all detection as unmatched if no tracks existing
    if (tracks.empty())
    {
        for (const auto &det : detection)
        {
            unmatched_det.push_back(det);
        }
        return;
    }

    std::vector<std::vector<float>> iou_matrix;
    // resize IOU matrix based on number of detection and tracks
    iou_matrix.resize(detection.size(), std::vector<float>(tracks.size()));

    std::vector<std::vector<float>> association;
    // resize association matrix based on number of detection and tracks
    association.resize(detection.size(), std::vector<float>(tracks.size()));


    // row - detection, column - tracks
    for (size_t i = 0; i < detection.size(); i++)
    {
        size_t j = 0;
        for (const auto &trk : tracks)
        {
            iou_matrix[i][j] = CalculateIou(detection[i], trk.second);
            j++;
        }
    }

    // Find association
    HungarianMatching(iou_matrix, detection.size(), tracks.size(), association);

    for (size_t i = 0; i < detection.size(); i++)
    {
        bool matched_flag = false;
        size_t j = 0;
        for (const auto &trk : tracks)
        {
            if (0 == association[i][j])
            {
                // Filter out matched with low IOU
                if (iou_matrix[i][j] >= iou_threshold)
                {
                    matched[trk.first] = detection[i];
                    matched_flag = true;
                }
                // It builds 1 to 1 association, so we can break from here
                break;
            }
            j++;
        }
        // if detection cannot match with any tracks
        if (!matched_flag)
        {
            unmatched_det.push_back(detection[i]);
        }
    }
}

/*
int TRACK::run(string &picPath)
{
    _frameCNT = (_frameCNT < kMinHits) ? _frameCNT + 1 : _frameCNT;

    auto ret = forward(picPath);
    if (ret != 0)
    {
        cerr << "forward error" << endl;
        return ret;
    }
    vector<xsk::Rect> bboxes_CV;
    for (auto &bbox : bboxes)
        bboxes_CV.emplace_back(bbox.xmin, bbox.ymin, bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin);
    for (auto &track : tracks)
        track.second.Predict();
    map<int, xsk::Rect> matched;
    vector<xsk::Rect> unmatched_det;
    if (!bboxes.empty())
        AssociateDetectionsToTrackers(bboxes_CV, tracks, matched, unmatched_det);
    for (const auto &match : matched)
    {
        const auto &ID = match.first;
        tracks[ID].Update(match.second);
    }
    for (const auto &det_ : unmatched_det)
    {
        Tracker tracker;
        tracker.Init(det_);
        // Create new track and generate new ID
        tracks[_currentID] = tracker;
        _currentID = _currentID < 999 ? _currentID + 1 : 0;
    }
    for (auto it = tracks.begin(); it != tracks.end();)
    {
        if (it->second.coast_cycles_ > kMaxCoastCycles)
            it = tracks.erase(it);
        else
            it++;
    }

    bboxes_track.clear();
    for (const auto &trk : tracks)
    {
        if (trk.second.coast_cycles_ < 1 and (trk.second.hit_streak_ >= kMinHits or _frameCNT < kMinHits))
        {
            const auto &bbox = trk.second.GetStateAsBbox();
            auto xmin = bbox.x;
            auto ymin = bbox.y;
            auto xmax = bbox.x + bbox.width;
            auto ymax = bbox.y + bbox.height;
            bboxes_track.emplace_back(BBOX_TRACK{trk.first, xmin, ymin, xmax, ymax});
        }
    }

    return 0;
}

int TRACK::run(unsigned char *imgPtr, size_t dataLen)
{
    _frameCNT = (_frameCNT < kMinHits) ? _frameCNT + 1 : _frameCNT;

    auto ret = forward(reinterpret_cast<float *>(imgPtr), dataLen);
    if (ret != 0)
    {
        cerr << "forward error" << endl;
        return ret;
    }
    vector<xsk::Rect> bboxes_CV;
    for (auto &bbox : bboxes)
        bboxes_CV.emplace_back(bbox.xmin, bbox.ymin, bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin);
    for (auto &track : tracks)
        track.second.Predict();
    map<int, xsk::Rect> matched;
    vector<xsk::Rect> unmatched_det;
    if (!bboxes.empty())
        AssociateDetectionsToTrackers(bboxes_CV, tracks, matched, unmatched_det);
    for (const auto &match : matched)
    {
        const auto &ID = match.first;
        tracks[ID].Update(match.second);
    }
    for (const auto &det_ : unmatched_det)
    {
        Tracker tracker;
        tracker.Init(det_);
        // Create new track and generate new ID
        tracks[_currentID] = tracker;
        _currentID = _currentID < 999 ? _currentID + 1 : 0;
    }
    for (auto it = tracks.begin(); it != tracks.end();)
    {
        if (it->second.coast_cycles_ > kMaxCoastCycles)
            it = tracks.erase(it);
        else
            it++;
    }

    bboxes_track.clear();
    for (const auto &trk : tracks)
    {
        if (trk.second.coast_cycles_ < 1 and (trk.second.hit_streak_ >= kMinHits or _frameCNT < kMinHits))
        {
            const auto &bbox = trk.second.GetStateAsBbox();
            auto xmin = bbox.x;
            auto ymin = bbox.y;
            auto xmax = bbox.x + bbox.width;
            auto ymax = bbox.y + bbox.height;
            bboxes_track.emplace_back(BBOX_TRACK{trk.first, xmin, ymin, xmax, ymax});
        }
    }
//    cout << "bboxes size = " << bboxes.size() << endl;
//    cout << "bboxes_track size = " << bboxes_track.size() << endl;
    return 0;
}

int TRACK::run(cv::Mat &img)
{
    _frameCNT = (_frameCNT < kMinHits) ? _frameCNT + 1 : _frameCNT;

    auto ret = forward(img);
    if (ret != 0)
    {
        cerr << "forward error" << endl;
        return ret;
    }
    vector<xsk::Rect> bboxes_CV;
    for (auto &bbox : bboxes)
        bboxes_CV.emplace_back(bbox.xmin, bbox.ymin, bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin);
    for (auto &track : tracks)
        track.second.Predict();
    map<int, xsk::Rect> matched;
    vector<xsk::Rect> unmatched_det;
    if (!bboxes.empty())
        AssociateDetectionsToTrackers(bboxes_CV, tracks, matched, unmatched_det);
    for (const auto &match : matched)
    {
        const auto &ID = match.first;
        tracks[ID].Update(match.second);
    }
    for (const auto &det_ : unmatched_det)
    {
        Tracker tracker;
        tracker.Init(det_);
        // Create new track and generate new ID
        tracks[_currentID] = tracker;
        _currentID = _currentID < 999 ? _currentID + 1 : 0;
    }
    for (auto it = tracks.begin(); it != tracks.end();)
    {
        if (it->second.coast_cycles_ > kMaxCoastCycles)
            it = tracks.erase(it);
        else
            it++;
    }

    bboxes_track.clear();
    for (const auto &trk : tracks)
    {
        if (trk.second.coast_cycles_ < 1 and (trk.second.hit_streak_ >= kMinHits or _frameCNT < kMinHits))
        {
            const auto &bbox = trk.second.GetStateAsBbox();
            auto xmin = bbox.x;
            auto ymin = bbox.y;
            auto xmax = bbox.x + bbox.width;
            auto ymax = bbox.y + bbox.height;
            bboxes_track.emplace_back(BBOX_TRACK{trk.first, xmin, ymin, xmax, ymax});
        }
    }

    return 0;
}
*/

TRACK::TRACK() = default;

TRACK::~TRACK() = default;

int TRACK::track(vector<DETECT::DET_BOX> &detBoxes)
{
    _frameCNT = (_frameCNT < kMinHits) ? _frameCNT + 1 : _frameCNT;

    vector<xsk::Rect> bboxes_CV;
    for (auto &bbox : detBoxes)
        bboxes_CV.emplace_back(bbox.xmin, bbox.ymin, bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin);
    for (auto &track : tracks)
        track.second.Predict();
    map<int, xsk::Rect> matched;
    vector<xsk::Rect> unmatched_det;
    if (!bboxes_CV.empty())
        AssociateDetectionsToTrackers(bboxes_CV, tracks, matched, unmatched_det);
    for (const auto &match : matched)
    {
        const auto &ID = match.first;
        tracks[ID].Update(match.second);
    }
    for (const auto &det_ : unmatched_det)
    {
        Tracker tracker;
        tracker.Init(det_);
        // Create new track and generate new ID
        tracks[_currentID] = tracker;
        _currentID = _currentID < 999 ? _currentID + 1 : 0;
        cout << "检查ID！！" << endl;
        cout << _currentID << endl;
    }
    for (auto it = tracks.begin(); it != tracks.end();)
    {
        if (it->second.coast_cycles_ > kMaxCoastCycles)
            it = tracks.erase(it);
        else
            it++;
    }
    trkBoxes.clear();
    for (const auto &trk : tracks)
    {
        if (trk.second.coast_cycles_ < 1 and (trk.second.hit_streak_ >= kMinHits or _frameCNT < kMinHits))
        {
            const auto &bbox = trk.second.GetStateAsBbox();
            auto xmin = bbox.x;
            auto ymin = bbox.y;
            auto xmax = bbox.x + bbox.width;
            auto ymax = bbox.y + bbox.height;
            trkBoxes.emplace_back(TRK_BOX{trk.first, label, xmin, ymin, xmax, ymax});
        }
    }

    return 0;
}

//COUNTER::~COUNTER()
//{
//    delete line;
//}
//
//// set the line position
//// xy are relative coordinates
//int COUNTER::reset(int oriWidth, int oriHeight, float x1, float y1, float x2, float y2, COUNT_MODE countMode_)
//{
//    delete line;
//    if (x1 < 0 or x1 > 1 or
//        y1 < 0 or y1 > 1 or
//        x2 < 0 or x2 > 1 or
//        y2 < 0 or y2 > 1)
//    {
//        cerr << "Invalid coordinates of the line." << endl;
//        return -1;
//    }
//
//
//    if (countMode_ != BOTTOM_LINE_CENTER and countMode_ != BOX_CENTER)
//    {
//        cerr << "Internal error, wrong countMode" << endl;
//        return -1;
//    }
//    countMode = countMode_;
//
//    return 0;
//}
//
