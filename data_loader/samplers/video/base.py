class Base:
  def __init__(self, video, label):
    self.video = video
    self.label = label

  def next(self, cache_frames):
    frames, frame_num = self.video.get_next_frames(cache=cache_frames)
    if not frames:
      return None, None, None
    targets = self.label.targets(frame_num)
    return frames, frame_num, targets

  def change_position(self):
    self.video.set_start_frame()
