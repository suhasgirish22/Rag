// Static content: the sample document, suggested questions, distractor
// chunks for experiments, and all quiz definitions.

export const SAMPLE_DOC = `Honeybees are social insects that live together in colonies of up to sixty thousand individuals. A single colony behaves almost like one organism: food gathering, temperature control, and defense are all shared tasks. Bees evolved alongside flowering plants over millions of years, and the partnership shaped both: flowers advertise with color and scent, and bees collect the nectar and pollen those flowers offer.

When a forager bee discovers a rich patch of flowers, she returns to the hive and performs the famous waggle dance. The dance is a figure-eight movement in which the angle of the waggle run tells other bees the direction of the food relative to the sun, and the duration of the waggle tells them the distance. Through this dance, bees communicate the location of food sources with remarkable precision, recruiting nestmates to the best patches.

Inside the hive there are three kinds of bees. The queen is the only fertile female and can lay over a thousand eggs in a single day. Worker bees are sterile females that do every other job: nursing larvae, building comb, guarding the entrance, and foraging. Drones are males whose single purpose is to mate with a new queen, after which they die.

Honey is made from nectar that foragers carry home in a special honey stomach. House bees pass the nectar from mouth to mouth, adding enzymes that break down its sugars, then deposit it into wax cells. The bees fan the cells with their wings to evaporate water until the nectar thickens into honey, and finally seal each cell with a wax cap for storage.

Honeybee populations face serious threats. The varroa mite, a parasite that attaches to bees and spreads viruses, is considered the single most damaging enemy of the western honeybee. Certain pesticides, especially neonicotinoids, can impair bee navigation and memory even at doses that do not kill. Habitat loss removes the diverse flowers that colonies need to stay healthy through the seasons.

The economic importance of bees is enormous. About one third of the food humans eat depends on insect pollination, and honeybees do much of that work. Almonds, apples, blueberries, and cucumbers are nearly impossible to grow at commercial scale without managed honeybee hives, which beekeepers truck from farm to farm as crops come into bloom.`;

export const SAMPLE_QUESTIONS = [
  "How do bees communicate the location of food?",
  "What threats do honeybee populations face?",
  "How is honey made from nectar?",
];

// Off-topic chunks for the "add irrelevant chunks" experiment.
export const DISTRACTOR_CHUNKS = [
  "Neapolitan pizza dough uses only four ingredients: flour, water, salt, and yeast. The dough must rest for at least eight hours, and the oven should reach four hundred eighty degrees Celsius so the crust blisters in about ninety seconds.",
  "The stock market closed mixed on Tuesday as technology shares rallied while energy stocks declined. Analysts pointed to quarterly earnings reports and shifting interest rate expectations as the main drivers of trading volume.",
  "To parallel park, pull up beside the car in front of the space, reverse slowly while turning the wheel toward the curb, then straighten out and counter-steer as your front wheels pass the other car's bumper.",
];

export const QUIZZES = {
  1: {
    question: "Why does a chatbot need RAG to answer questions about your private documents?",
    options: [
      { text: "Because the model was never trained on your documents, so it has no way to know what's in them", correct: true },
      { text: "Because private documents are encrypted and models can't read encryption", correct: false },
      { text: "Because models can only answer questions about topics from the last year", correct: false },
    ],
    explain: "A language model only 'knows' what was in its training data. RAG works around this by retrieving passages from your documents and showing them to the model at question time — like an open-book exam.",
  },
  2: {
    question: "What's the main downside of making chunks very large?",
    options: [
      { text: "Each chunk mixes several topics, so its match to any one specific question gets fuzzier", correct: true },
      { text: "Large chunks take longer to alphabetize", correct: false },
      { text: "Large chunks can't be stored in a database", correct: false },
    ],
    explain: "A huge chunk covering five topics is only ~20% about any one of them, so it matches a specific question weakly — and it drags irrelevant text into the prompt. Small chunks are precise but risk cutting an idea in half. That's the chunking trade-off.",
  },
  3: {
    question: "What does it mean when two chunks appear close together in semantic space?",
    options: [
      { text: "Their meanings are similar, even if they use different words", correct: true },
      { text: "They appear next to each other in the original document", correct: false },
      { text: "They contain the same number of words", correct: false },
    ],
    explain: "Embeddings place text by meaning, not by position or length. Two chunks about the same topic land close together even if they come from opposite ends of the document and share few exact words.",
  },
  4: {
    question: "Which parts of the pipeline were already done before the user asked anything?",
    options: [
      { text: "Chunking the document and embedding the chunks", correct: true },
      { text: "Retrieving the top chunks and generating the answer", correct: false },
      { text: "Nothing — the whole pipeline runs only after a question arrives", correct: false },
    ],
    explain: "Chunking and embedding are 'indexing' — done once, ahead of time, and stored in a vector database. Only embedding the question, retrieval, and generation happen live when a question arrives. That's what makes RAG fast.",
  },
  5: {
    question: "Why must the question be embedded with the same method as the chunks?",
    options: [
      { text: "So it lands in the same space, making 'find relevant text' as simple as 'find the nearest points'", correct: true },
      { text: "Because questions are stored in the database next to the chunks", correct: false },
      { text: "It doesn't matter — any embedding method works for the question", correct: false },
    ],
    explain: "Distances are only meaningful between points on the same map. If the question were embedded differently, 'near' and 'far' would be meaningless, and retrieval would return nonsense.",
  },
  6: {
    question: "What does a cosine similarity score of 0.05 between a chunk and the question suggest?",
    options: [
      { text: "The chunk is almost certainly irrelevant to the question", correct: true },
      { text: "The chunk is a 5% match, which is usually good enough to use", correct: false },
      { text: "The chunk contains exactly 5 matching words", correct: false },
    ],
    explain: "Cosine similarity runs from 0 (unrelated) to 1 (same direction in meaning-space). Scores near zero mean the chunk and question have almost nothing in common, so retrieval skips it.",
  },
  7: {
    question: "What exactly does the language model receive in a RAG system?",
    options: [
      { text: "One plain-text prompt containing instructions, the retrieved chunks, and the question", correct: true },
      { text: "A live connection to the vector database so it can search by itself", correct: false },
      { text: "The embedding vectors of the retrieved chunks", correct: false },
    ],
    explain: "There's no magic link between the model and your database. The retrieved chunks are pasted as ordinary text into the prompt — that's the entire interface. The model never sees the embeddings.",
  },
  8: {
    question: "Why do good RAG answers cite which chunk supports each claim?",
    options: [
      { text: "So you can verify the answer against the source instead of trusting the model blindly", correct: true },
      { text: "Because models legally must cite sources", correct: false },
      { text: "Citations make the answer use fewer tokens", correct: false },
    ],
    explain: "Grounding plus attribution is RAG's superpower: every claim can be traced to a real passage. If a sentence has no supporting chunk, that's a red flag the model may be making it up (hallucinating).",
  },
  9: {
    question: "Your RAG system gives a wrong answer. Based on your experiments, what's the FIRST thing to check?",
    options: [
      { text: "Whether retrieval actually surfaced chunks that contain the answer", correct: true },
      { text: "Whether the language model needs more training", correct: false },
      { text: "Whether the answer was generated too quickly", correct: false },
    ],
    explain: "As you saw when you removed the key chunk: if retrieval doesn't surface the right text, no model can answer correctly. Most RAG failures are retrieval failures — check what was retrieved before blaming the model.",
    continueLabel: "Finish the lesson 🎓",
  },
};
