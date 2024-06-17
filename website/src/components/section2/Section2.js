import React from "react";

const CenterWrapper = (props) => {
    return (
        <section className="section">
            <div className="container is-max-desktop">
                <div className="columns is-centered has-text-centered">
                    <div className="column is-four-fifths">
                        {props.content}
                    </div>
                </div>
            </div>
        </section>
    );
}

const Content = () => {
    return (
        <div>
        <h2 className="title is-3">DiffCom is Robust to Unexpected Transmission Degradations</h2>
        <img id="method"
             height={"100%"}
             src={process.env.PUBLIC_URL + "/imgs/Fig_generalization.png"}
             alt={"loading.."}/>
        <p> A visual comparison illustrating the impact of several unexpected
            transmission degradations: ① unseen channel fading, ② PAPR reduction, ③
            with ISI (removed CP symbols), and ④ very low CSNR (0dB). </p>
        </div>
    )
}

const Section2 = () => {
    return (
        <CenterWrapper content={<Content />}/>
    );
}

export default Section2
