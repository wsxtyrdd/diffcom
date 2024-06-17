import {Button} from "@mui/material";
import React, {Component} from "react";
import {VscGithub} from "react-icons/vsc"
import {FaFilePdf} from "react-icons/fa"
import {SiArxiv} from "react-icons/si"

const AuthorBlock = (props) => (
    <span className="author-block">
        <a href={props.link}>{props.name}</a>
        <sup>{props.number}</sup>,
    </span>
)

const LinkButton = (props) => (
    <Button sx={{m: '0.3rem'}}
            style={{
                borderRadius: 35,
                backgroundColor: "black",
                padding: "0.5rem 1.0rem"
            }}
            href={props.link}
            variant="contained"
            startIcon={props.icon}>
        {props.text}
    </Button>
);

export default class Header extends Component {
    render() {
        return (
            <section className="hero information">
                <div className="container is-max-desktop">
                    <div className="columns is-centered">
                        <div className="column has-text-centered">
                            <h1 className="title is-1 publication-title">
                                DiffCom: Channel Received Signal is a Natural Condition to Guide Diffusion Posterior
                                Sampling
                            </h1>
                            <div className="is-size-5 publication-authors">
                                <AuthorBlock name="Sixian Wang"
                                             link="https://scholar.google.com/citations?user=f9s8H6UAAAAJ&hl=zh-CN&oi=ao"
                                /> <AuthorBlock name="Jincheng Dai"
                                                link="https://scholar.google.com/citations?hl=zh-CN&user=0I_YtFsAAAAJ"
                            /> <span className="author-block">Kailin Tan,
                            </span> <AuthorBlock name="Xiaoqi Qin"
                                                 link="https://scholar.google.com/citations?user=mrEeosAAAAAJ&hl=zh-CN"
                            /> <AuthorBlock name="Kai Niu"
                                            link="https://scholar.google.com/citations?user=Dm9tNxoAAAAJ&hl=zh-CN"
                            /> <span className="author-block">Ping Zhang, </span>
                            </div>
                            <div className="is-size-5 publication-authors">
                                <span className="author-block">Beijing University of Posts and Telecommunications (BUPT), Beijing, China</span>
                            </div>
                            {/*Publication links*/}
                            <div className="column has-text-centered">
                                {/*<LinkButton link={"."} icon={<FaFilePdf/>} text="Paper"/>*/}
                                <LinkButton link={"https://arxiv.org/pdf/2406.07390"} icon={<SiArxiv/>} text="arXiv"/>
                                <LinkButton link={"https://github.com/wsxtyrdd/diffcom"} icon={<VscGithub/>}
                                            text="Code"/>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        );
    }
}
