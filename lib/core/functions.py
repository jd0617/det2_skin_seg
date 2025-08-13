# from detectron2.engine import DefaultTrainer
# from detectron2.checkpount import DetectionCheckpointer



# class TrainerWithDiceBCELoss(DefaultTrainer):
#     @classmethod
#     def run_step(self):
#         assert self.model.training

#         data = next(self._data_loader_iter)
#         output = self.model(data)

#         gt_masks = [x["instance"].gt_masks.tensor.to(self.model.device) for x in data]
#         loss = self.model(data)["loss"]

        



