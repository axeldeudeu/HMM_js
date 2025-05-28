/**
 * Modèle de Markov Caché (HMM) pour la prédiction de caractères
 * Version interactive avec sélection de caractère par position
 */

import * as R from 'ramda';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { createInterface } from 'readline';

const keyboardLayout = {
    'a': ['z', 'q', 's'], 'z': ['a', 'e', 's', 'd'], 'e': ['z', 'r', 'd', 'f'],
    'r': ['e', 't', 'f', 'g'], 't': ['r', 'y', 'g', 'h'], 'y': ['t', 'u', 'h', 'j'],
    'u': ['y', 'i', 'j', 'k'], 'i': ['u', 'o', 'k', 'l'], 'o': ['i', 'p', 'l', 'm'],
    'p': ['o', 'm'], 'q': ['a', 'w', 's'], 's': ['q', 'a', 'z', 'd', 'w', 'x'],
    'd': ['s', 'z', 'e', 'f', 'x', 'c'], 'f': ['d', 'e', 'r', 'g', 'c', 'v'],
    'g': ['f', 'r', 't', 'h', 'v', 'b'], 'h': ['g', 't', 'y', 'j', 'b', 'n'],
    'j': ['h', 'y', 'u', 'k', 'n', ','], 'k': ['j', 'u', 'i', 'l', ',', ';'],
    'l': ['k', 'i', 'o', 'm', ';', ':'], 'm': ['l', 'o', 'p', ':', '!'],
    'w': ['q', 's', 'x'], 'x': ['w', 's', 'd', 'c'], 'c': ['x', 'd', 'f', 'v'],
    'v': ['c', 'f', 'g', 'b'], 'b': ['v', 'g', 'h', 'n'], 'n': ['b', 'h', 'j', ','],
    ',': ['n', 'j', 'k', ';'], ';': ['k', 'l', ':'], ':': ['l', 'm', '!'],
    '!': ['m', ':'], ' ': [' ']
};

const preprocessText = R.pipe(
    R.toLower,
    R.replace(/[^\w\sàáâãäåçèéêëìíîïñòóôõöùúûüýÿ]/g, ' '),
    R.replace(/\s+/g, ' '),
    R.trim
);

const extractUniqueChars = R.pipe(
    R.split(''),
    R.uniq,
    R.filter(R.complement(R.isEmpty))
);

const countCharacters = R.pipe(
    R.split(''),
    R.countBy(R.identity)
);

const createCharacterPairs = R.pipe(
    R.split(''),
    R.aperture(2),
    R.map(([a, b]) => [a, b])
);

const extractSentences = R.pipe(
    R.split(/[.!?]+/),
    R.map(R.trim),
    R.filter(sentence => sentence.length > 15),
    R.filter(R.complement(R.isEmpty))
);

const selectRandomSentence = R.pipe(
    sentences => {
        const randomIndex = Math.floor(Math.random() * sentences.length);
        return sentences[randomIndex];
    }
);

const buildTransitionCounts = R.pipe(
    R.groupBy(R.head),
    R.map(R.pipe(
        R.map(R.last),
        R.countBy(R.identity)
    ))
);

const addLaplaceSmoothing = (states) => (transitionCounts) =>
    R.map(counts => {
        const baseState = R.zipObj(states, R.repeat(1, states.length));
        return R.mergeWith(R.add, baseState, counts || {});
    })(transitionCounts);

const normalizeTransitions = R.map(counts => {
    const total = R.sum(R.values(counts));
    return R.map(count => count / total, counts);
});

const buildEmissionMatrix = (states) => {
    const createEmissionForState = (state) => {
        const baseEmission = R.zipObj(states, R.repeat(0.01, states.length));
        const correctEmission = R.assoc(state, 0.8, baseEmission);

        if (keyboardLayout[state]) {
            const adjacentKeys = keyboardLayout[state];
            const errorProb = 0.15 / adjacentKeys.length;
            const adjacentEmissions = R.pipe(
                R.filter(key => R.includes(key, states)),
                R.map(key => [key, errorProb]),
                R.fromPairs
            )(adjacentKeys);
            return R.mergeWith(R.add, correctEmission, adjacentEmissions);
        }
        return correctEmission;
    };

    return R.pipe(
        R.map(createEmissionForState),
        R.map(emissions => {
            const total = R.sum(R.values(emissions));
            return R.map(prob => prob / total, emissions);
        })
    )(R.zipObj(states, states));
};
class InteractiveHMM {
    constructor() {
        this.states = [];
        this.transitionMatrix = {};
        this.emissionMatrix = {};
        this.initialProbs = {};
        this.characterCounts = {};
        this.totalCharacters = 0;
        this.trainingHistory = [];
        this.modelPath = 'hmm_model.json';
    }
    loadExistingModel() {
        if (existsSync(this.modelPath)) {
            try {
                const modelData = JSON.parse(readFileSync(this.modelPath, 'utf8'));
                this.states = modelData.states || [];
                this.transitionMatrix = modelData.transitionMatrix || {};
                this.emissionMatrix = modelData.emissionMatrix || {};
                this.initialProbs = modelData.initialProbs || {};
                this.characterCounts = modelData.characterCounts || {};
                this.totalCharacters = modelData.totalCharacters || 0;
                this.trainingHistory = modelData.trainingHistory || [];
                console.log(`✓ Modèle HMM chargé (${this.trainingHistory.length} entraînements précédents)`);
                return true;
            } catch (error) {
                console.log("⚠ Erreur lors du chargement du modèle, création d'un nouveau modèle");
                return false;
            }
        }
        return false;
    }
    saveModel() {
        const modelData = {
            states: this.states,
            transitionMatrix: this.transitionMatrix,
            emissionMatrix: this.emissionMatrix,
            initialProbs: this.initialProbs,
            characterCounts: this.characterCounts,
            totalCharacters: this.totalCharacters,
            trainingHistory: this.trainingHistory,
            lastUpdate: new Date().toISOString()
        };

        try {
            writeFileSync(this.modelPath, JSON.stringify(modelData, null, 2));
            console.log("✓ Modèle HMM sauvegardé avec succès");
        } catch (error) {
            console.log("⚠ Erreur lors de la sauvegarde du modèle");
        }
    }
    learnFromText(text) {
        const processedText = preprocessText(text);
        const newStates = extractUniqueChars(processedText);
        const newCharCounts = countCharacters(processedText);

        const learningPipeline = R.pipe(
            R.tap(() => {
                this.states = R.uniq([...this.states, ...newStates]);
            }),
            R.tap(() => {
                this.characterCounts = R.mergeWith(R.add, this.characterCounts, newCharCounts);
                this.totalCharacters += processedText.length;
            }),

            R.tap(() => {
                this.initialProbs = R.mapObjIndexed(
                    (count) => count / this.totalCharacters,
                    this.characterCounts
                );
            }),

            R.tap((text) => {
                this.improveTransitionMatrix(text);
            }),

            R.tap(() => {
                this.emissionMatrix = buildEmissionMatrix(this.states);
            }),

            R.tap(() => {
                this.trainingHistory.push({
                    timestamp: new Date().toISOString(),
                    textLength: processedText.length,
                    newStates: newStates.length,
                    totalStates: this.states.length
                });
            })
        );

        learningPipeline(processedText);
        this.saveModel();
        return this;
    }
    improveTransitionMatrix(text) {
        const newPairs = createCharacterPairs(text);
        const newTransitionCounts = buildTransitionCounts(newPairs);

        const mergedCounts = R.mergeWith(
            (oldCounts, newCounts) => R.mergeWith(R.add, oldCounts || {}, newCounts || {}),
            this.transitionMatrix,
            newTransitionCounts
        );

        this.transitionMatrix = R.pipe(
            addLaplaceSmoothing(this.states),
            normalizeTransitions
        )(mergedCounts);
    }

    predictNextCharacter(inputChar, topN = 3) {
        if (!R.includes(inputChar, this.states)) {
            return {
                error: `Caractère '${inputChar}' non trouvé dans le modèle`,
                predictions: []
            };
        }

        const transitions = R.pathOr({}, [inputChar], this.transitionMatrix);

        const calculatePredictions = R.pipe(
            R.toPairs,
            R.map(([nextChar, transitionProb]) => ({
                character: nextChar,
                transitionProb,
                emissionProb: R.pathOr(0.8, [nextChar, nextChar], this.emissionMatrix),
                combinedScore: transitionProb * R.pathOr(0.8, [nextChar, nextChar], this.emissionMatrix)
            })),
            R.filter(pred => pred.transitionProb > 0),
            R.sortBy(R.prop('transitionProb')),
            R.reverse,
            R.take(topN),
            R.map(pred => ({
                character: pred.character,
                probability: `${(pred.transitionProb * 100).toFixed(4)}%`,
                emissionProb: `${(pred.emissionProb * 100).toFixed(4)}%`,
                combinedScore: `${(pred.combinedScore * 100).toFixed(6)}%`
            }))
        );

        return {
            inputCharacter: inputChar,
            predictions: calculatePredictions(transitions),
            totalPredictions: R.keys(transitions).length,
            modelVersion: this.trainingHistory.length
        };
    }

    getModelInfo() {
        return {
            numberOfStates: this.states.length,
            totalCharacters: this.totalCharacters,
            trainingIterations: this.trainingHistory.length,
            lastTraining: R.last(this.trainingHistory)?.timestamp || 'Jamais'
        };
    }
}
class CharacterSelector {
    constructor(sentence, hmm) {
        this.sentence = sentence;
        this.hmm = hmm;
        this.rl = createInterface({
            input: process.stdin,
            output: process.stdout
        });
    }

    displaySentenceWithPositions() {
        console.log(`\n Phrase sélectionnée :`);
        console.log(`"${this.sentence}"`);
        console.log(`\nLongueur : ${this.sentence.length} caractères\n`);

        console.log(` Caractères disponibles :`);
        R.pipe(
            R.split(''),
            R.addIndex(R.map)((char, index) => {
                const displayChar = char === ' ' ? '[ESPACE]' : char;
                console.log(`   Position ${index.toString().padStart(2, '0')}: '${displayChar}'`);
            })
        )(this.sentence);
    }

    async selectCharacterInteractive() {
        return new Promise((resolve) => {
            const askForPosition = () => {
                this.rl.question('\n Choisissez une position (0-' + (this.sentence.length - 1) + ') ou "q" pour quitter : ', (answer) => {
                    if (answer.toLowerCase() === 'q') {
                        console.log('\n Au revoir !');
                        this.rl.close();
                        resolve(null);
                        return;
                    }

                    const position = parseInt(answer);

                    if (isNaN(position) || position < 0 || position >= this.sentence.length) {
                        console.log(` Position invalide. Veuillez entrer un nombre entre 0 et ${this.sentence.length - 1}`);
                        askForPosition();
                        return;
                    }

                    const selectedChar = this.sentence[position];
                    console.log(`\n Caractère sélectionné à la position ${position} : '${selectedChar === ' ' ? '[ESPACE]' : selectedChar}'`);

                    const result = this.hmm.predictNextCharacter(selectedChar, 3);

                    if (result.error) {
                        console.log(` ${result.error}`);
                    } else {
                        console.log(`\n Top 3 probabilités pour le caractère suivant :`);
                        console.log(`${'='.repeat(60)}`);

                        result.predictions.forEach((pred, index) => {
                            const displayChar = pred.character === ' ' ? '[ESPACE]' : pred.character;
                            console.log(`   ${index + 1}. '${displayChar}' → ${pred.probability}`);
                        });

                        if (position + 1 < this.sentence.length) {
                            const actualNext = this.sentence[position + 1];
                            const actualPred = result.predictions.find(p => p.character === actualNext);

                            console.log(`\n Vérification :`);
                            if (actualPred) {
                                console.log(`    Caractère réel suivant '${actualNext === ' ' ? '[ESPACE]' : actualNext}' prédit avec ${actualPred.probability}`);
                            } else {
                                console.log(`    Caractère réel suivant '${actualNext === ' ' ? '[ESPACE]' : actualNext}' non dans le top 3`);
                            }
                        } else {
                            console.log(`\n Fin de phrase atteinte, pas de caractère suivant.`);
                        }
                    }

                    console.log(`\n${'='.repeat(60)}`);
                    askForPosition();
                });
            };

            askForPosition();
        });
    }
}
async function runInteractiveHMM() {
    console.log("=== HMM Interactif - Sélection de Caractère par Position ===\n");

    const loadTextData = R.pipe(
        () => {
            const files = ['mots_generes.txt', 'mots_filtrés.txt'];
            for (const file of files) {
                if (existsSync(file)) {
                    console.log(`✓ Chargement du fichier '${file}'`);
                    return readFileSync(file, 'utf8');
                }
            }
            console.log("⚠ Aucun fichier trouvé, utilisation du texte par défaut");
            return `
            L'intelligence artificielle transforme notre monde moderne.
            Les algorithmes d'apprentissage automatique analysent les données.
            La reconnaissance vocale permet la communication naturelle.
            Les réseaux de neurones imitent le cerveau humain.
            La programmation fonctionnelle utilise des pipes élégants.
            Les modèles statistiques prédisent les comportements futurs.
            `;
        }
    );

    try {
        const text = loadTextData();
        const sentences = extractSentences(text);
        const selectedSentence = selectRandomSentence(sentences);

        const hmm = new InteractiveHMM();
        hmm.loadExistingModel();
        hmm.learnFromText(text);

        const info = hmm.getModelInfo();
        console.log(`\n Modèle HMM :`);
        console.log(`   - États (caractères) : ${info.numberOfStates}`);
        console.log(`   - Entraînements : ${info.trainingIterations}`);

        const selector = new CharacterSelector(selectedSentence, hmm);
        selector.displaySentenceWithPositions();

        console.log(`\nInstructions :`);
        console.log(`   - Choisissez une position pour voir les probabilités du caractère suivant`);
        console.log(`   - Le modèle affichera les 3 meilleures prédictions`);
        console.log(`   - Tapez "q" pour quitter`);

        await selector.selectCharacterInteractive();

    } catch (error) {
        console.error("Erreur dans le pipeline :", error.message);
    }
}
export {
    InteractiveHMM,
    CharacterSelector,
    runInteractiveHMM,
    keyboardLayout,
    preprocessText,
    extractUniqueChars,
    extractSentences,
    selectRandomSentence
};

if (process.argv[1] === new URL(import.meta.url).pathname) {
    runInteractiveHMM();
}