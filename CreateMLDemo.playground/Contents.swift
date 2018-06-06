import CreateMLUI
import Foundation
import CreateML

// Image Classification
let builder = MLImageClassifierBuilder()
builder.showInLiveView()

// Text Classification
let data = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/SaiKambampati/Desktop/spam.json"))
let (trainingData, testingData) = data.randomSplit(by: 0.8, seed: 5)
let spamClassifier = try MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "label")
let trainingAccuracy = (1.0 - spamClassifier.trainingMetrics.classificationError) * 100
let validationAccuracy = (1.0 - spamClassifier.validationMetrics.classificationError) * 100
let evaluationMetrics = spamClassifier.evaluation(on: testingData)
let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100
let metadata = MLModelMetadata(author: "Sai Kambampati", shortDescription: "A model trained to classify spam messages", version: "1.0")
try spamClassifier.write(to: URL(fileURLWithPath: "/Users/SaiKambampati/Desktop/SpamDetector.mlmodel"), metadata: metadata)

// Table Classification
let houseData = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/SaiKambampati/Desktop/HouseData.csv"))
let (trainingCSVData, testCSVData) = houseData.randomSplit(by: 0.8, seed: 0)

let pricer = try MLRegressor(trainingData: houseData, targetColumn: "MEDV")

let csvMetadata = MLModelMetadata(author: "Sai Kambampati", shortDescription: "A model used to determine the price of a house based on some features.", version: "1.0")
try pricer.write(to: URL(fileURLWithPath: "/Users/SaiKambampati/Desktop/HousePricer.mlmodel"), metadata: csvMetadata)
