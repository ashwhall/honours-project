import random

class Sequential:
  '''
  Samples frames from videos sequentially, for up to max_consecutive_frames
  '''
  def __init__(self, video_samplers, max_consecutive_frames, cache_frames, keep_video_file_open):
    self.samplers = video_samplers
    self.finished_samplers = []
    random.shuffle(self.samplers)
    self.curr_sampler_index = 0
    self.max_consecutive_frames = max_consecutive_frames
    self.cache_frames = cache_frames
    self.keep_video_file_open = keep_video_file_open
    self.generator = self._next_generator()

  def _next_generator(self):
    while True:
      sampler = self.samplers[self.curr_sampler_index]
      for _ in range(self.max_consecutive_frames):
        frames, frame_num, targets = sampler.next(cache_frames=True)
        if frames is not None:
          yield (frames, frame_num, targets, sampler.label)
        else:
          # pretend this video never happened
          self.finished_samplers.append(self.samplers.pop(self.curr_sampler_index))
          self.curr_sampler_index -= 1
          break
      if not self.cache_frames:
        sampler.video.clear_cache()
      if not self.keep_video_file_open:
        sampler.video.close_file()

      # If we have no more unfinished samplers left, start through again
      if len(self.samplers) == 0:
        tmp_samplers = self.samplers
        self.samplers = self.finished_samplers
        self.finished_samplers = tmp_samplers
        random.shuffle(self.samplers)
        self.curr_sampler_index = 0
      else:
        self.curr_sampler_index = (self.curr_sampler_index + 1) % len(self.samplers)

  def next(self):
    return next(self.generator)

  def __len__(self):
    return len(self.samplers)

  def __getitem__(self, indices):
    return self.samplers[indices]
