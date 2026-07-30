import unittest

class AdaptiveFrameSampler:
    def __init__(self, min_fps=10, max_fps=30, stable_time=90, movement_threshold=0.15, confidence_threshold=0.60):
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.stable_time = stable_time
        self.movement_threshold = movement_threshold
        self.confidence_threshold = confidence_threshold
        
        self.last_motion_time = None
        self.current_fps = self.max_fps

    def update(self, movement_score, confidence, current_time):
        if self.last_motion_time is None:
            self.last_motion_time = current_time
        trigger_flag = False
        if confidence < self.confidence_threshold:
            trigger_flag = True

        if movement_score > self.movement_threshold:
            self.last_motion_time = current_time 
            self.current_fps = self.max_fps      
        else:
            time_stable = current_time - self.last_motion_time
            
           
            if time_stable >= self.stable_time:           
                self.current_fps = self.min_fps           
            elif time_stable >= (self.stable_time * 0.75): 
                self.current_fps = 15
            elif time_stable >= (self.stable_time * 0.50): 
                self.current_fps = 20
            elif time_stable >= (self.stable_time * 0.25): 
                self.current_fps = 25

       
        return {
            "fps": self.current_fps,
            "trigger_yolo": trigger_flag
        }



class TestAdaptiveFrameSampler(unittest.TestCase):
    
    def test_stable_decay(self):
        sampler = AdaptiveFrameSampler()
        sampler.update(movement_score=0.1, confidence=0.9, current_time=0)
        
        result = sampler.update(movement_score=0.1, confidence=0.9, current_time=95)
        
        self.assertEqual(result["fps"], sampler.min_fps)
        self.assertFalse(result["trigger_yolo"])

    def test_movement_spike(self):
        sampler = AdaptiveFrameSampler()
        sampler.update(movement_score=0.1, confidence=0.9, current_time=0)
        
        result = sampler.update(movement_score=0.9, confidence=0.9, current_time=50)
        
        self.assertEqual(result["fps"], sampler.max_fps)
        self.assertFalse(result["trigger_yolo"]) 

    def test_trigger_yolo(self):
        sampler = AdaptiveFrameSampler()
        sampler.update(movement_score=0.1, confidence=0.9, current_time=0)
        
        result = sampler.update(movement_score=0.1, confidence=0.1, current_time=30)
        
        self.assertTrue(result["trigger_yolo"])
        
    def test_hysteresis_no_oscillation(self):
        sampler = AdaptiveFrameSampler()
        sampler.update(movement_score=0.1, confidence=0.9, current_time=0)
        
        result = sampler.update(movement_score=0.1, confidence=0.9, current_time=50)
        
        self.assertEqual(result["fps"], 20)

if __name__ == '__main__':
    unittest.main()