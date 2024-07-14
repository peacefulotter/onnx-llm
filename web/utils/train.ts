import * as ort from 'onnxruntime-web/training'
import { SortData } from './data'

const SESSSION_PATH = './_next/static/chunks/pages/'

const getTrainingSession = async (): Promise<ort.TrainingSession> => {
    const createOptions: ort.TrainingSessionCreateOptions = {
        checkpointState: 'checkpoint',
        trainModel: 'training_model.onnx',
        evalModel: 'eval_model.onnx',
        optimizerModel: 'optimizer_model.onnx',
    }

    for (const key in createOptions) {
        createOptions[key as keyof typeof createOptions] = `${SESSSION_PATH}${
            createOptions[key as keyof typeof createOptions]
        }`
    }

    console.log('Creating training session with options: ', createOptions)

    try {
        const session = await ort.TrainingSession.create(createOptions, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
        })
        return session
    } catch (err) {
        console.log('Error loading the training session: ' + err)
        throw err
    }
}

export async function runTraining(
    max_iters: number = 1000,
    blockSize: number = 11,
    vocabSize: number = 3,
    batchSize: number = 2
) {
    const session = await getTrainingSession()
    const dataset = new SortData(session, blockSize, vocabSize, batchSize)
    let iter = 0

    const lossName = session.trainingOutputNames[0]
    const data = []

    while (iter < max_iters) {
        const start = Date.now()

        const batch = dataset.getTrainingBatch()
        const results = await session.runTrainStep(batch)

        const dt = Date.now() - start
        const loss = (results[lossName].data as Float32Array)[0]

        if (iter % 25 === 0) {
            console.log(`Training iteration ${iter} took ${dt}ms, loss: ${loss}`)
        }
        data.push({ iter, loss, dt })

        // update weights then reset gradients
        await session.runOptimizerStep()
        await session.lazyResetGrad()

        ++iter
    }

    console.log(data)

    await session.release()
}
