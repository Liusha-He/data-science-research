from typing import List, Tuple

import numpy as np
from scipy.special import softmax


class ATRIUM(object):
    params = {
        "gamma_rule": 0.87080,
        "gamma_gate": 1.0,
        "beta_rule": -4.5,
        "beta_gate": -1.78984,
        "specificity": 1.28296,
        "phi": 4.07742,
        "lrate_exemplar": 0.32163,
        "lrate_rule": 0.03375,
        "lrate_gate": 0.41313,
        "lrate_attention": 1.96593,
        "cost_e": 1,
        "cost_r": 1,
    }

    weights = {
        "exemplar": np.random.rand(4, 10, 10),
        "rule": np.random.rand(2, 4),
        "gate": np.random.rand(10, 10),
        "attention": np.array([0.5, 0.5]),
    }

    coordinates = [
        np.tile(np.array(range(10)), (10, 1)).T,
        np.tile(np.array(range(10)), (10, 1)),
    ]

    def activate(self, stimulus: Tuple):
        """This method produces output response probabilities based on learned weights.

        :param stimulus:
        :param coordinates:

        :return:

        """
        a_large = (
            1
            + np.exp(
                -self.params["gamma_rule"] * (stimulus[0] + self.params["beta_rule"])
            )
        ) ** -1
        a_small = 1 - a_large

        a_rules = np.array([a_large, a_small])
        rule_response_activation = a_rules.reshape([1, 2]) @ self.weights["rule"]

        m = np.zeros(self.coordinates[0].shape)

        ds = []

        for i, h in enumerate(stimulus):
            d = np.abs(h - self.coordinates[i])
            ds.append(d)
            m += self.weights["attention"][i] * d

        a_exemplars = np.exp(-0.5 * self.params["specificity"] * m)

        n = len(self.weights["exemplar"])
        r = []

        for i in range(n):
            r.append(np.sum(np.multiply(a_exemplars, self.weights["exemplar"][i])))

        exemplar_response_activation = np.array(r)

        a_gate = np.sum(np.multiply(self.weights["gate"], a_exemplars))
        a_gate = (
            1 + np.exp(-self.params["gamma_gate"] * a_gate + self.params["beta_gate"])
        ) ** -1

        output = {
            "a_rules": a_rules,
            "a_gate": a_gate,
            "a_exemplars": a_exemplars,
            "a_r": rule_response_activation,
            "a_e": exemplar_response_activation,
            "distance": ds,
            "preds": a_gate * softmax(self.params["phi"] * exemplar_response_activation)
            + (1 - a_gate) * softmax(self.params["phi"] * rule_response_activation),
        }

        return output

    def fit(self, stimuli: List[Tuple], labels: List[int], n_epoch: int) -> np.ndarray:

        outp = []

        for n in range(n_epoch):
            for i, stimulus in enumerate(stimuli):
                y = [0] * 4
                y[labels[i] - 1] = 1

                preds = self.activate(stimulus)

                print(
                    f"{stimulus} -> actual category {labels[i]} -> "
                    f"predicted {np.argmax(preds['preds']) + 1} ({preds['preds'][0][labels[i] - 1]:.3f})"
                )

                outp.append(preds["preds"] * np.array(y))

                preds["a_r"][preds["a_r"] > 1] = 1
                preds["a_r"][preds["a_r"] < 0] = 0
                preds["a_e"][preds["a_e"] > 1] = 1
                preds["a_e"][preds["a_e"] < 0] = 0
                preds["preds"][preds["preds"] > 1] = 1
                preds["preds"][preds["preds"] < 0] = 0

                # changes
                RA = np.exp(
                    -0.5
                    * self.params["cost_r"]
                    * np.linalg.norm(np.array(y) - preds["a_r"])
                )
                EA = np.exp(
                    -0.5
                    * self.params["cost_e"]
                    * np.linalg.norm(np.array(y) - preds["a_e"])
                )
                MA = preds["a_gate"] * EA + (1 - preds["a_gate"]) * RA

                # rule weights
                delta_wr = (
                    self.params["lrate_rule"]
                    * ((1 - preds["a_gate"]) * RA * self.params["cost_r"] / MA)
                    * (np.array(y) - preds["a_r"])
                )
                delta_wr = np.reshape(preds["a_rules"], (2, 1)) @ np.reshape(
                    delta_wr, (1, 4)
                )

                # exemplar weights
                delta_we_1 = (
                    self.params["lrate_exemplar"]
                    * (preds["a_gate"] * EA * self.params["cost_e"] / MA)
                    * (np.array(y) - preds["a_e"])
                )
                delta_we = []
                for delta in delta_we_1:
                    delta_we.append(delta * preds["a_exemplars"])

                delta_we = np.array(delta_we)

                # attention weights
                delta_watt_1 = (preds["a_gate"] * EA * self.params["cost_e"] / MA) * (
                    np.array(y) - preds["a_e"]
                )
                delta_watt_2 = np.zeros(self.weights["exemplar"][0].shape)

                for i, e in enumerate(delta_watt_1):
                    delta_watt_2 += e * self.weights["exemplar"][i]

                delta_watt_2 = (
                    delta_watt_2 * preds["a_exemplars"] * self.params["specificity"]
                )

                delta_watt = []
                for d in preds["distance"]:
                    delta_watt.append(
                        -self.params["lrate_attention"] * (np.sum(d * delta_watt_2))
                    )

                delta_watt = np.array(delta_watt)

                # gate weights
                delta_wg = (
                    self.params["lrate_gate"]
                    * ((EA - RA) / MA)
                    * (1 - preds["a_gate"])
                    * preds["a_gate"]
                    * self.params["gamma_gate"]
                )
                delta_wg = delta_wg * np.reshape(preds["a_exemplars"], (10, 10))

                # update
                self.weights["rule"] += delta_wr
                self.weights["exemplar"] += delta_we
                self.weights["gate"] += delta_wg
                self.weights["attention"] += delta_watt

        return np.array(outp)

    def evaluate(self, stimuli: List[Tuple], labels: List[int], n_epoch: int) -> None:
        outp = self.fit(stimuli, labels, n_epochs)  # noqa


if __name__ == "__main__":
    n_epochs = 29
    stimuli = [
        (2, 2),
        (2, 2),
        (7, 7),
        (7, 7),
        (5, 1),
        (5, 8),
        (6, 4),
        (7, 2),
        (8, 5),
        (4, 1),
        (4, 8),
        (3, 5),
        (2, 7),
        (1, 4),
    ]
    labels = [3, 3, 4, 4, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    model = ATRIUM()

    model.evaluate(stimuli, labels, n_epochs)
