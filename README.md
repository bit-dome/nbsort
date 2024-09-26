# nbsort

Simple object tracer

- IoU box tracker
- with optical-flow for fix camera movement





## Usage

```
# detections = np.array([x1,y1,x2,y2,score], [x1,y1,x2,y2,score], ...)
online_targets = tracker.update_tracks(detections, image_frame)
```
