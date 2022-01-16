# import unittest
# import losses.focal_loss as fl

# # boiler plate code.. create a class
# # inherit for run function from base class
# class Test_TestFocalLoss(unittest.TestCase):
#     def test_negativeBeta(self):
        
#         # https://stackoverflow.com/questions/129507/how-do-you-test-that-a-python-function-throws-an-exception
#         with self.assertRaises(Exception) as msg:
#             fl.reweight([10],-1)
#             fl.reweight([10],1.1)
#         print(msg.exception)
#     def test_zero(self):
#         self.assertEqual(fl.reweight([1,2,3],0),[1,1,1])
#     def test_99(self):
#         ans=[0,0,0]
#         ans[0]=(1-0.99)/(1-(0.99**1))
#         ans[1]=(1-0.99)/(1-(0.99**2))
#         ans[2]=(1-0.99)/(1-(0.99**3))
#         self.assertEqual(fl.reweight([1,2,3],0.99),ans)