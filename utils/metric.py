
import numpy as np

def load_buckets(total, buckets):
   bucket_list = [0] # keep buckt range [1, 4, 8] ==> (1,4] (4,8]
   if buckets == 0:
       return [0], [0]
   bucket_size = total//buckets
   buckets_counter = [0] * (buckets+1) # [0, 0, 0, 0, 0] add 1 for 0
   for i in range(1, total+1):
       if i % bucket_size == 0:
           bucket_list.append(i)
   return bucket_list, buckets_counter

def discount_factors(buckets):
   discount_factor = np.arange(0, 1, 1/(buckets+1)).tolist()
   return discount_factor

def DCA(pred_list, tgt_list, bucket_num):
   bucket_size = 1
   bucket_num = bucket_num

   # initialization
   bucket_list, counter = load_buckets(bucket_size*bucket_num, bucket_num)
   discount_factor = discount_factors(bucket_num)
   assert len(bucket_list) == len(discount_factor), (len(bucket_list), len(discount_factor))

   # count occurences of diff value in each bucket
   for tgt, pred in zip(tgt_list, pred_list):
       diff = abs(tgt - pred)
       for i, upper_bound in enumerate(bucket_list):
           if diff <= upper_bound:
               counter[i] += 1
               break
   # print(discount_factor)

   # calculate discounted accuracy
   p_b = np.array(counter)/len(pred_list) # indicator/total
   discounted_accuracy = np.sum(p_b * (1 - np.array(discount_factor)))

   return discounted_accuracy