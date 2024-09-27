



# def LossBalancing(list_previous_losses: list, avg_combined_loss: list, priority:list, beta=0):
#     # Losses are in the list of list form
#     lossFuncCount = len(list_previous_losses)
#     epoch = len(list_previous_losses[0])
#     # beta for epoch>=2
#     new_weights=[i/sum(priority) for i in priority]
#     difficulty = [1 for i in range(lossFuncCount)]
    
#     if(epoch >= 1): #scale balancing
#         old1_combined_loss = avg_combined_loss[-1]
#         for index,i in enumerate(list_previous_losses):
#             new_weights[index]=priority[index]*old1_combined_loss/i[-1]
        
    
#     if(epoch >=2): # difficulty
#         old1_combined_loss = avg_combined_loss[-1]
#         old2_combined_loss = avg_combined_loss[-2]
#         print(f'Last2TotalLosses: {[old2_combined_loss,old1_combined_loss]}')
#         print(f'Last2HeightLosses: {[list_previous_losses[0][-2],list_previous_losses[0][-1]]}')
#         print(f'Last2footPrintLosses: {[list_previous_losses[1][-2],list_previous_losses[1][-1]]}')
        
#         for index,i in enumerate(list_previous_losses):
#             difficulty[index] = (i[-1]/i[-2])/(old1_combined_loss/old2_combined_loss)
#         difficulty = [i**beta for i in difficulty]
#     return new_weights, difficulty



# if(epoch>=2):
#     alpha =  1/sum([i*j for i,j in zip(difficulty,priority)])
#     avg_tloss = alpha *(height_tloss + footprint_tloss)
# else:
#     avg_tloss = (height_tloss + footprint_tloss)


class LossBalancer:
    def __init__(self, taskPriority, beta=1):
        self.beta = beta
        self.priority = taskPriority
        self.weights = {i:j for i,j in taskPriority.items()}
        self.difficulty = {i:1 for i in taskPriority.keys()}
        self.AVGLoss = []
        
    def computeLoss(self,individualLoss):
        loss=0
        for task in individualLoss.keys():
            loss+=self.weights[task]*self.difficulty[task]*individualLoss[task]
        return loss
    
    def computeAVGLoss(self, trainingLoss): #computing Average Loss
        epoch = len(self.AVGLoss)+1
        summ=0
        for i in trainingLoss.values():
            summ+=i[-1]
        self.AVGLoss.append(summ)
        
        if(epoch >=2):
            summ = 0
            for task in self.difficulty.keys():
                summ+=self.difficulty[task]*self.priority[task]
            alpha =  1/summ
            self.AVGLoss[-1]*=alpha
            
        
    def update(self, trainingLoss):
        self.computeAVGLoss(trainingLoss)
        epoch = len(self.AVGLoss)
        
        if(epoch >= 1): #weights updation
            for task in self.weights.keys():
                self.weights[task] = self.priority[task]*self.AVGLoss[-1]/trainingLoss[task][-1]

        if(epoch >=2): # difficulty updation
            for task in self.difficulty.keys():
                self.difficulty[task] = (trainingLoss[task][-1]/trainingLoss[task][-2])/(self.AVGLoss[-1]/self.AVGLoss[-2])
                self.difficulty[task] = self.difficulty[task]**self.beta