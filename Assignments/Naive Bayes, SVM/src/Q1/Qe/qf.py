a=[[8585, 1415],
 [ 840, 4160]]




# positive examples
precision=a[0][0]/(a[0][0]+a[0][1])
recall=a[0][0]/(a[0][0]+a[1][0])
f1=(2*precision*recall)/(precision+recall)
print("Positive model")
print("Precision: "+str(precision))
print("Recall: "+str(recall))
print("F1-score: "+str(f1))

# negative examples
precision=a[1][1]/(a[1][1]+a[1][0])
recall=a[1][1]/(a[1][1]+a[0][1])
f1=(2*precision*recall)/(precision+recall)
print("Negative model")
print("Precision: "+str(precision))
print("Recall: "+str(recall))
print("F1-score: "+str(f1))



