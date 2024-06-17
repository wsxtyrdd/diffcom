import React, {Fragment} from "react";

const OverviewBlock = () => (
    <section className="hero overview">
        <div className="container is-max-desktop">
            <img id="overview"
                 height={"100%"}
                 src={process.env.PUBLIC_URL + "/imgs/overview.png"}
                 alt={"overview"}>
            </img>
            <h2 className="subtitle has-text-centered">
                Overview of the <span className="ddiffcom">DiffCom</span> system architecture, and overall concept of
                the proposed method.
            </h2>
        </div>
    </section>
)

const AbstactBlock = () => (
    <section className="section">
        <div className="container is-max-desktop">
            <div className="columns is-centered has-text-centered">
                <div className="column is-four-fifths">
                    <h2 className="title is-2">Abstract</h2>
                    <div className="content has-text-justified">
                        <p>
                            End-to-end visual communication systems typically optimize a trade-off between channel
                            bandwidth
                            costs and signal-level distortion metrics.
                            However, under challenging physical conditions, such a discriminative
                            communication paradigm often results in unrealistic reconstructions with perceptible
                            blurring and aliasing artifacts, despite the inclusion of perceptual or adversarial losses
                            during training. This issue primarily stems from the receiver's limited knowledge about the
                            underlying data
                            manifold and the use of deterministic decoding mechanisms.
                        </p>
                        <p>
                            We propose <span className="ddiffcom">DiffCom</span>, a novel end-to-end <span
                            className="bold">generative communication paradigm that utilizes off-the-shelf
                            generative priors from diffusion models for decoding </span>, thereby improving perceptual
                            quality
                            without heavily relying on bandwidth costs and received signal quality.
                            Unlike traditional systems that rely on deterministic decoders optimized solely for
                            distortion
                            metrics, our <span className="ddiffcom">DiffCom</span> leverages <span className="bold"> raw channel-received signal as a fine-grained condition to guide stochastic posterior
                        sampling.</span> Our approach ensures that reconstructions remain on the manifold of real data
                            with a
                            novel confirming constraint, enhancing the robustness and reliability of the generated
                            outcomes.
                            Furthermore, <span className="ddiffcom">DiffCom</span> incorporates a blind posterior
                            sampling
                            technique to address
                            scenarios with unknown forward transmission characteristics.
                        </p>
                        <p>
                            Experimental results demonstrate that:
                            <ul>
                                <li><span className="ddiffcom">DiffCom</span> achieves SOTA transmission performance in
                                    terms of
                                    multiple perceptual quality metrics, such as LPIPS, DISTS, FID, and so on.
                                </li>
                                <li><span className="ddiffcom">DiffCom</span> significantly enhances the robustness of
                                    current
                                    methods against various transmission-related degradations, including mismatched SNR,
                                    unseen
                                    fading, blind channel estimation, PAPR reduction, and inter-symbol interference.
                                </li>
                            </ul>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </section>
)

const Section1 = () => {
    return (
        <Fragment>
            <br/>
            <OverviewBlock/>
            <AbstactBlock/>
        </Fragment>
    );
}

export default Section1;
