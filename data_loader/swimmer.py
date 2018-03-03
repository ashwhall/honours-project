from helper import Helper

class Swimmer:
  '''
  A single swimmer, essentially composed of a list of key-frames
  '''
  def __init__(self, lane_number, keyframes):
    self.lane_number = lane_number
    self.keyframes = keyframes

  def get_interpolated_coords(self, at_frame):
    '''
    Returns the swimmer coordinates at the given at_frame
    '''
    before_keyframe = None
    after_keyframe = None
    for keyframe in self.keyframes:
      # There is a keyframe exactly at this frame_number
      if keyframe['frame'] == at_frame:
        return [keyframe['x'], keyframe['y']]

      # We found the frame before the desired frame
      if keyframe['frame'] < at_frame:
        if keyframe['stop_frame']:
          before_keyframe = None
          continue
        before_keyframe = keyframe

      # We found the frame after the desired frame
      elif keyframe['frame'] > at_frame:
        if before_keyframe is None:
          return None
        else:
          after_keyframe = keyframe

      # We have before and after keyframes -> return their interpolated coords
      if before_keyframe is not None and after_keyframe is not None:
        return Swimmer._interpolate_coords(before_keyframe, after_keyframe, at_frame)

    # We can use the swimmer's last-known coordinates
    if before_keyframe is not None:
      return [before_keyframe['x'], before_keyframe['y']]

  @staticmethod
  def _interpolate_coords(before_keyframe, after_keyframe, at_frame):
    '''
    Linearly interpolates to `at_frame` between the two given keyframes
    '''
    new_x = Helper.linear_interpolate(before_keyframe['frame'],
                                      after_keyframe['frame'],
                                      before_keyframe['x'],
                                      after_keyframe['x'],
                                      at_frame)
    new_y = Helper.linear_interpolate(before_keyframe['frame'],
                                      after_keyframe['frame'],
                                      before_keyframe['y'],
                                      after_keyframe['y'],
                                      at_frame)
    return [new_x, new_y]
