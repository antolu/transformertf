from transformertf.models.phylstm import LossWeights


class TestLossWeights:
    #  Tests that a default instance of LossWeights is created with the correct default values
    def test_default_instance(self) -> None:
        lw = LossWeights()
        assert lw.alpha == 1.0
        assert lw.beta == 1.0
        assert lw.gamma == 1.0
        assert lw.eta == 1.0
        assert lw.kappa == 1.0

    #  Tests that a custom instance of LossWeights is created with the correct values
    def test_custom_instance(self) -> None:
        lw = LossWeights(alpha=2.0, beta=3.0, gamma=4.0, eta=5.0, kappa=6.0)
        assert lw.alpha == 2.0
        assert lw.beta == 3.0
        assert lw.gamma == 4.0
        assert lw.eta == 5.0
        assert lw.kappa == 6.0

    #  Tests that a LossWeights instance with alpha=0.0 is created correctly
    def test_alpha_zero(self) -> None:
        lw = LossWeights(alpha=0.0)
        assert lw.alpha == 0.0
        assert lw.beta == 1.0
        assert lw.gamma == 1.0
        assert lw.eta == 1.0
        assert lw.kappa == 1.0

    #  Tests that a LossWeights instance with beta=0.0 is created correctly
    def test_beta_zero(self) -> None:
        lw = LossWeights(beta=0.0)
        assert lw.alpha == 1.0
        assert lw.beta == 0.0
        assert lw.gamma == 1.0
        assert lw.eta == 1.0
        assert lw.kappa == 1.0

    #  Tests that a LossWeights instance with gamma=0.0 is created correctly
    def test_gamma_zero(self) -> None:
        lw = LossWeights(gamma=0.0)
        assert lw.alpha == 1.0
        assert lw.beta == 1.0
        assert lw.gamma == 0.0
        assert lw.eta == 1.0
        assert lw.kappa == 1.0

    #  Tests that a LossWeights instance with eta=0.0 is created correctly
    def test_eta_zero(self) -> None:
        lw = LossWeights(eta=0.0)
        assert lw.alpha == 1.0
        assert lw.beta == 1.0
        assert lw.gamma == 1.0
        assert lw.eta == 0.0
        assert lw.kappa == 1.0

    #  Tests that creating an instance of LossWeights with kappa=0.0 sets the kappa attribute to 0.0
    def test_kappa_zero(self) -> None:
        lw = LossWeights(kappa=0.0)
        assert lw.kappa == 0.0

    #  Tests that an instance of LossWeights is created with beta=2.0
    def test_create_instance_with_beta_2(self) -> None:
        lw = LossWeights(beta=2.0)
        assert lw.beta == 2.0

    #  Tests that the eta value is set correctly
    def test_eta_value(self) -> None:
        lw = LossWeights(eta=2.0)
        assert lw.eta == 2.0

    #  Tests that the gamma value is set correctly
    def test_gamma_value(self) -> None:
        lw = LossWeights(gamma=2.0)
        assert lw.gamma == 2.0

    #  Tests that an instance of LossWeights can be created with kappa=2.0
    def test_create_instance_with_kappa_2(self) -> None:
        lw = LossWeights(kappa=2.0)
        assert lw.kappa == 2.0
