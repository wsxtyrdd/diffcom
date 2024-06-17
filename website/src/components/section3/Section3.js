import React, {useState} from "react";
import {Button, Grid, Stack, ToggleButton, ToggleButtonGroup} from '@mui/material';
import ReactSwipe from 'react-swipe'
import {ReactCompareSlider, ReactCompareSliderImage} from 'react-compare-slider';
import {AiFillLeftCircle, AiFillRightCircle} from 'react-icons/ai'

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

const IamgeComareSlider = ({imgs}) => {
    return (
        <ReactCompareSlider
            itemOne={<ReactCompareSliderImage src={imgs.input} alt='input image'/>}
            itemTwo={<ReactCompareSliderImage src={imgs.recon} alt='recon image'/>}
        />
    );
}

const Carousel = ({images, kernels, task, method, index, onButton}) => {
    let reactSwipeEl;

    const nextIndex = (index, change, length) => {
        let next_idx = (index + change);
        if (next_idx < 0) {
            next_idx = length + next_idx;
        } else {
            next_idx = next_idx % length;
        }
        return next_idx;
    }

    const pushPrev = () => {
        reactSwipeEl.prev();
        onButton(nextIndex(index, -1, images.length));
    }

    const pushNext = () => {
        reactSwipeEl.next();
        onButton(nextIndex(index, 1, images.length));
    }

    return (
        <Grid container direction="column" style={{margin: '1.5rem 0 0 0'}}>
            <Grid container direction="row">
                {/*<Grid item xs={2} md={2} sm={2}>*/}
                {/*</Grid>*/}
                <Grid item xs={8} md={8} sm={8}>
                    {/*<p style={{margin: '0 1rem 0 0', fontWeight: 'bold'}}>Slide the button for comparison</p>*/}
                    <p style={{margin: '0 1rem 0 0', fontWeight: 'bold'}}>Left: HiFi-DiffCom (with {method} Encoder); Right: {method}</p>
                    <ReactSwipe
                        className="carousel"
                        swipeOptions={{continuous: true, disableScroll: true}}
                        ref={el => (reactSwipeEl = el)}
                        childCount={images.length}
                    >
                        {images.map((image_pair) => {
                            return (
                                <div>
                                    <IamgeComareSlider imgs={image_pair}/>
                                </div>
                            );
                        })}
                    </ReactSwipe>
                </Grid>
                <Grid item xs={4} md={4} sm={4}>
                    <div style={{margin: '0 0 0 1.24rem'}}>
                        <GridKernel task={task} kernels={kernels[index]}/>
                    </div>
                </Grid>
            </Grid>
            <Grid item xs style={{margin: '1rem 0 0 0'}}>
                <Stack justifyContent="center" alignItems="center" direction="row" spacing={2}>
                    <Button startIcon={<AiFillLeftCircle/>} variant={"outlined"}
                            onClick={() => pushPrev()}> Prev </Button>
                    <Button endIcon={<AiFillRightCircle/>} variant={"outlined"}
                            onClick={() => pushNext()}> Next </Button>
                </Stack>
            </Grid>
        </Grid>
    );
}

const GridKernel = ({kernels}) => {
    return (
        <Grid item>
            <Grid item>
                <Stack direction="column" style={{display: 'flex'}}>
                    <p style={{fontSize: "1rem", fontWeight: "bold", margin: 0}}>VTM + 5G LDPC</p>
                    <img id="method"
                         src={kernels.recon}
                         alt={"loading.."}/>
                </Stack>
            </Grid>
            <Grid item>
                <Stack direction="column" style={{display: 'flex'}}>
                    <p style={{fontSize: "1rem", fontWeight: "bold", margin: '1rem 0 0 0'}}>Original Image</p>
                    <img id="method"
                         src={kernels.truth}
                         alt={"loading.."}/>
                </Stack>
            </Grid>
        </Grid>
    );
}

function range(start, end) {
    let array = [];
    for (let i = start; i < end; i++) {
        array.push(i);
    }
    return array;
}

const ImageDisplay = ({method}) => {
    const task = 'SNR1';
    const [index, setIndex] = useState(0);

    const images = range(0, 3).map((idx) => {
        return ({
            'input': process.env.PUBLIC_URL + '/imgs/results/' + method + '/' + task + '/input_' + idx + '.png',
            'recon': process.env.PUBLIC_URL + '/imgs/results/' + method + '/' + task + '/recon_' + idx + '.png',
        });
    })

    const kernels = range(0, 3).map((idx) => {
        return ({
            'recon': process.env.PUBLIC_URL + '/imgs/results/' + method + '/' + task + '/vtm_' + idx + '.png',
            'truth': process.env.PUBLIC_URL + '/imgs/results/' + method + '/' + task + '/ori_' + idx + '.png',
        });
    });

    return (
        <Carousel images={images} kernels={kernels} task={task} method={method} index={index} onButton={setIndex}/>
    )
}


const Content = () => {
    // const task_pair = {
    //     'SNR1': '0dB',
    //     'SNR2': '10dB'
    // }
    const method_pair = {
        'DeepJSCC': 'DeepJSCC',
        'NTSCC': 'NTSCC'
    };

    // const tasks = ['SNR1', 'SNR2'];
    // const [task, setTask] = useState('SNR1');

    const methods = ['DeepJSCC', 'NTSCC'];
    const [method, setMethod] = useState('DeepJSCC');

    // const onTaskToggle = (button_val) => {
    //     setTask(button_val);
    // };

    const onMethodToggle = (button_val) => {
        setMethod(button_val);
    };

    return (
        <div>
            <h2 className="title is-3">DiffCom Exhibits Superior Transmission Performance</h2>
            <ToggleButtonGroup
                color="primary"
                value={method}
                size="small"
                aria-label="Platform">
                <p>
                    <span className="ddiffcom">HiFi-DiffCom vs. </span> &nbsp;
                </p>
                {methods.map(t => (
                    <ToggleButton value={t} onClick={() => {
                        onMethodToggle(t)
                    }} id={t} key={t}>
                        {method_pair[t]}
                    </ToggleButton>))
                }
            </ToggleButtonGroup>
            , AWGN channel, CSNR = 0dB, CBR = 1/48;
            <ImageDisplay method={method}/>
        </div>
    );
}

const Section3 = () => {
    return (
        <CenterWrapper content={<Content/>}/>
    );
}

export default Section3
