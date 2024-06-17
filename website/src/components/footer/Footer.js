import React, {Component} from "react";
import { IconButton } from "@mui/material";
import { VscGithub } from "react-icons/vsc"
import {FaFilePdf} from "react-icons/fa"

const LinkButton = (props) => (
    <IconButton href={props.link}>
        {props.icon}
  </IconButton>
);

export default class Footer extends Component{
  render(){
      return (
        <footer className="footer">
        <div className="container">
            <div className="content has-text-centered">
            <LinkButton link={"https://arxiv.org/abs/"} icon={<FaFilePdf />} text="Paper"/>
            <LinkButton link={"https://github.com/wsxtyrdd/diffcom"} icon={<VscGithub />} text="Code"/>
            </div>
            <div className="columns is-centered">
            <div className="column is-8">
                <div className="content">
                <p>
                    This website is licensed under a <a rel="license"
                                                        href="http://creativecommons.org/licenses/by-sa/4.0/">Creative
                    Commons Attribution-ShareAlike 4.0 International License</a>.
                </p>
                <p>
                    This means you are free to borrow the <a
                    href="https://github.com/BlindDPS/diffcom-project">source code</a> of this website,
                    we just ask that you link back to this page in the footer.
                    Please remember to remove the analytics code included in the header of the website which
                    you do not want on your website.
                </p>
                </div>
            </div>
            </div>
        </div>
        </footer>
    );
  }
}
