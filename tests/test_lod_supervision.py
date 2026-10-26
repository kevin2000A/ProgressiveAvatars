import unittest
from utils.lod_supervision import level_loss_weights

class SupervisionTests(unittest.TestCase):
    def test_schedule_and_control(self):
        base=[1/31,2/31,4/31,8/31,16/31]
        self.assertEqual(level_loss_weights(5,.5,200000,'anneal',200000,250000),base)
        self.assertEqual(level_loss_weights(5,.5,300000,'multi',200000,250000),base)
        self.assertEqual(level_loss_weights(5,.5,250000,'anneal',200000,250000),[0.,0.,0.,0.,1.])
        middle=level_loss_weights(5,.5,225000,'anneal',200000,250000)
        for i in range(4):self.assertAlmostEqual(middle[i],base[i]/2)
        self.assertAlmostEqual(middle[-1],base[-1]/2+.5)
        for n in [1,4,5]:
            for step in [180000,200000,225000,250000,300000]:
                weights=level_loss_weights(n,.5,step,'anneal',200000,250000)
                self.assertAlmostEqual(sum(weights),1.)
                self.assertTrue(all(w>=0 for w in weights))

if __name__=='__main__':unittest.main()
