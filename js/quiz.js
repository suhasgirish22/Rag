// Tiny quiz component. Renders one multiple-choice question into a container.
// The learner can answer (right or wrong — feedback either way), then a
// "Continue" button unlocks the next step.

export function renderQuiz(container, quiz, onContinue) {
  container.innerHTML = "";
  const box = document.createElement("div");
  box.className = "quiz";
  box.innerHTML = `
    <p class="quiz-kicker">✅ Quick check</p>
    <h3>${quiz.question}</h3>
    <div class="quiz-options"></div>
    <div class="quiz-feedback" hidden></div>
  `;
  const optsEl = box.querySelector(".quiz-options");
  const feedbackEl = box.querySelector(".quiz-feedback");

  // Shuffle options so the right answer isn't always in the same spot
  const options = quiz.options
    .map((o, i) => ({ ...o, i }))
    .sort(() => Math.random() - 0.5);

  let answered = false;
  for (const opt of options) {
    const btn = document.createElement("button");
    btn.className = "quiz-option";
    btn.textContent = opt.text;
    btn.addEventListener("click", () => {
      if (answered) return;
      answered = true;
      for (const b of optsEl.querySelectorAll("button")) {
        b.disabled = true;
        const isCorrect = quiz.options[options.find(o => o.text === b.textContent).i].correct;
        if (isCorrect) b.classList.add("correct");
      }
      if (!opt.correct) btn.classList.add("wrong");

      feedbackEl.hidden = false;
      feedbackEl.className = `quiz-feedback ${opt.correct ? "ok" : "nope"}`;
      feedbackEl.textContent = opt.correct
        ? `✓ Correct! ${quiz.explain}`
        : `✗ Not quite. ${quiz.explain}`;

      if (onContinue) {
        const cont = document.createElement("button");
        cont.className = "btn btn-primary quiz-continue";
        cont.textContent = quiz.continueLabel || "Continue to the next step →";
        cont.addEventListener("click", () => onContinue());
        box.appendChild(cont);
      }
    });
    optsEl.appendChild(btn);
  }
  container.appendChild(box);
}
