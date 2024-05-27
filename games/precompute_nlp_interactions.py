"""This script pre-computes the interaction values for large NLP games with SVARMIQ and stores them in a pickle file."""
import pickle
import os
import time

from scipy.special import binom

from approximators import SvarmIQ, SHAPIQEstimator, KernelSHAPIQEstimator, PermutationSampling
from games import NLPGame

if __name__ == "__main__":
    # Parameters -----------------------------------------------------------------------------------
    BUDGET = 100_000  # 100_000
    ORDER = 2

    sentences_to_explain = [
        "Destined to be a classic before it was even conceptualized and This game deserves all the recognition it deserves",  # 20
        "I can not stand Ben stiller anymore How this man is allowed to still make movies is beyond me",  # 20
        "This outing of Knotts includes one of his best sidekicks ever Frank Welker who makes the",  # 20
        "Uwe Boll slips back in his film-making skills once again to offer up a scifi",  # 20
        "First let me just comment on what I liked about the movie The special effects were fantastic and very rarely",  # 20
        "I read the book before seeing the movie and the film is one of the very best adaptations out there",  # 20
        "OK I'm Italian but there aren't so many Italian film like this I think this one",  # 20
        "This movie pretty much sucked I'm in the Army and the soldiers depicted in this movie are horrible",  # 20
        "This movie basically is a very well made production and gives a good impression of a war situation and effects",  # 20
        "I remember watching this series avidly every Saturday evening It was the highlight of the week I loved everything",  # 20
        "Do you ever wonder what is the worst movie ever made stop wondering I'm telling you Michael is",  # 20
        "I really liked this movie I have seen several Gene Kelly flicks and this is one of his best",  # 20
        "I remember I saw this cartoon when I was 6 or 7 and I was really looking forward to this",  # 20
        "An awful film It must have been up against some real stinkers to be nominated for the Golden Globe",  # 20
        "After a very scary crude opening which gives you that creepy 'Chainsaw massacre' feeling everything falls apart",  # 20
    ]

    for sentence_to_explain in sentences_to_explain:
        sentence_id = "".join([word[0] for word in sentence_to_explain.split(" ")])
        sentence_id = sentence_id.replace("'", "")

        ESTIMATOR_NAME = "SVARM-IQ"  # "SVARM-IQ", "KernelSHAP-IQ", "SHAP-IQ" "Permutation"

        # Set up the game and get sentence and parts -----------------------------------------------
        game = NLPGame(input_text=sentence_to_explain, set_zero=True)
        n = game.n
        N = set(range(n))

        n_interactions = sum([binom(n, k) for k in range(1, ORDER + 1)])

        print(
            f"Input sentence: {sentence_to_explain}\n"
            f"Sentence ID: {sentence_id}\n"
            f"Number of tokens: {n}\n"
            f"Budget: {BUDGET}\n"
            f"# Interactions: {n_interactions}\n"
            f"Estimator: {ESTIMATOR_NAME}\n"
        )

        # Set up the estimator ---------------------------------------------------------------------
        start_time = time.time()
        if ESTIMATOR_NAME == "SVARM-IQ":
            estimator = SvarmIQ(
                N=N, order=ORDER, interaction_type="SII", top_order=False, replacement=True
            )
            sii_values = estimator.approximate_with_budget(game=game.set_call, budget=BUDGET)
        elif ESTIMATOR_NAME == "SHAP-IQ":
            estimator = SHAPIQEstimator(N=N, order=ORDER, interaction_type="SII", top_order=False)
            sii_values = estimator.compute_interactions_from_budget(
                game=game.set_call, budget=BUDGET
            )
        elif ESTIMATOR_NAME == "KernelSHAP-IQ":
            estimator = KernelSHAPIQEstimator(N=N, order=ORDER, interaction_type="SII")
            sii_values = estimator.approximate_with_budget(game_fun=game.set_call, budget=BUDGET)
        elif ESTIMATOR_NAME == "Permutation":
            estimator = PermutationSampling(
                N=N, order=ORDER, interaction_type="SII", top_order=False
            )
            sii_values = estimator.approximate_with_budget(game=game.set_call, budget=BUDGET)
        else:
            raise ValueError(f"Estimator {ESTIMATOR_NAME} not found.")

        estimator_time = time.time() - start_time
        print("Creation took: ", estimator_time, "seconds.")

        # Save the results -------------------------------------------------------------------------
        # define folder name and save path
        DATA_FOLDER = os.path.join("precomputed_nlp_interactions", ESTIMATOR_NAME)
        os.makedirs(DATA_FOLDER, exist_ok=True)

        # see how many files are already in the folder
        RUN_NUMBER = len(os.listdir(DATA_FOLDER)) + 1

        file_name = f"{sentence_id}_{n}_budget_{BUDGET}_order_{ORDER}_{RUN_NUMBER}" + ".pkl"
        file_path = os.path.join(DATA_FOLDER, file_name)

        stored_data = {
            "sentence": sentence_to_explain,
            "n": n,
            "budget": BUDGET,
            "sii_values": sii_values,
        }

        with open(file_path, "wb") as f:
            pickle.dump(stored_data, f)
        print(f"Saved results to {file_path}.")
